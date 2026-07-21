import os
import pickle
from multiprocessing import Pool
import shutil
from collections import defaultdict
import regex as re
from tqdm import tqdm
import numpy as np


def pretokenize(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

def count_pairs(ids, weight=1, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:  # 파일을 바이너리 모드로 열기
        # 파일 크기 확인
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks

        # 청크 시작 위치 계산(동일 간격)
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # 마지막에 파일 끝 위치 추가

        buffer_size = 4096  # 경계부터 미리 읽을 바이트 수

        # 경계 위치 조정(종료 토큰 검색)
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)  # 경계의 추정 위치부터 시작

            while True:
                buffer = file.read(buffer_size)  # 버퍼 크기만큼 읽기

                # 파일 끝에 도달한 경우
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # 읽어 온 청크에서 종료 토큰 검색
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # 발견한 경우 해당 위치를 새로운 경계로 설정
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                # 발견하지 못했다면 다음 버퍼 위치로 이동
                chunk_position += buffer_size

    # 중복을 제거하고 정렬하여 반환
    return sorted(set(chunk_boundaries))

def process_single_chunk(file_path, start, end, end_token):
    """청크 하나를 처리하는 함수"""
    pretoken_counts = defaultdict(int)

    # 파일을 열고 청크를 읽어 들임
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # 특수 토큰을 기준으로 분할
        texts = chunk_text.split(end_token)

        # 각 텍스트를 사전 토큰화
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    # 파일을 열고 청크를 읽어 들임
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # 특수 토큰을 기준으로 분할
        texts = chunk_text.split(end_token)

        # 각 텍스트를 사전 토큰화
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=8):
    # 청크 준비
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunks = len(chunk_boundaries) - 1

    chunk_info_list = []
    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    # 병렬 처리
    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), total=len(chunk_info_list), desc="Pretokenizing"))

    # 결과 통합
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    # 사전 토큰을 ID열로 변환
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}


    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)  # 캐시

    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):  # 캐시에 등록
            pair_to_ids[pair].add(ids)

    for step in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:  # 쌍이 존재하지 않으면 루프 종료
            break

        # 가장 자주 등장하는 쌍 선택
        # best_pair = max(pair_counts, key=pair_counts.get)
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # best_pair가 포함된 ID열을 캐시에서 가져옴
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]  # 더 이상 사용하지 않으므로 삭제

        # 영향을 받는 ID열만 갱신
        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)]              # 기존 ID열 삭제
            ids_counts[tuple(new_ids)] = ids_count  # 새로운 ID열 추가

            # 기존 쌍의 빈도 감소
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            # 새로운 쌍의 빈도 증가
            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))

    return merge_rules

class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath):
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))

        def get_merge_priority(pair):
            return self.merge_rules.get(pair, float('inf'))  # 병합 규칙에 없는 ID 쌍은 우선순위를 가장 낮게 설정

        while len(ids) > 1:
            # 현재 쌍을 가져옴
            counts = count_pairs(ids)

            # 우선순위가 가장 높은 쌍을 찾음
            best_pair = min(counts, key=get_merge_priority)

            # 병합 가능 여부 확인
            if best_pair not in self.merge_rules:
                break

            # 병합 수행
            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)

        return ids

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        # show_progress가 True이면 tqdm으로 진행 상황 표시
        texts = tqdm(texts) if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                # 각 사전 토큰을 BPE로 인코딩
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def _encode_chunk(self, args):
        """청크를 처리하여 디스크에 캐시"""
        file_path, start, end, cache_dir, chunk_idx = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            # 청크 인코딩
            ids = self.encode(chunk_text)

        # 캐시 파일에 저장
        cache_file = os.path.join(cache_dir, f"chunk_{chunk_idx:05d}.npy")
        np.array(ids, dtype=np.uint16).tofile(cache_file)

        return cache_file, len(ids)


    def encode_file(self, file_path, output_file,
                                    num_processes=4, num_chunks=64,
                                   cache_dir="bpe_cache"):

        # 캐시 디렉터리 준비
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # 청크를 병렬로 토큰화하여 캐시에 저장
            chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
            total_chunks = len(chunk_boundaries) - 1

            chunk_info_list = []
            for i in range(total_chunks):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                chunk_info_list.append((file_path, start, end, cache_dir, i))

            with Pool(processes=num_processes) as pool:
                cache_results = list(tqdm(
                    pool.imap(self._encode_chunk, chunk_info_list),
                    total=len(chunk_info_list),
                    desc="Encoding chunks"
                ))

            # 전체 토큰 수 계산
            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            total_tokens = sum(token_counts)

            # 메모리 맵 파일 생성
            dtype = np.uint16
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

            # 배치 단위로 캐시에서 메모리 맵에 쓰기
            # OpenWebText 예제처럼 배치 처리
            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            arr.flush()
            del arr

        finally:
            # 캐시 삭제
            shutil.rmtree(cache_dir)

        return total_tokens

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text