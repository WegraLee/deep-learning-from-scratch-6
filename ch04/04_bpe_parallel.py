import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
from multiprocessing import Pool
from collections import defaultdict
import regex as re
from tqdm import tqdm
import pickle


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

    with open(file_path, "rb") as file:

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks


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

def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    # 파일을 열고 청크를 읽어 들임
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # 특수 토큰 기준으로 분할
        texts = chunk_text.split(end_token)

        # 각 텍스트를 사전 토큰화
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=64):
    # 1단계: 청크 준비
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunks = len(chunk_boundaries) - 1

    chunk_info_list = []
    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    # 2단계: 병렬 처리로 사전 토큰화
    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), total=len(chunk_info_list), desc="Pretokenizing"))

    # 3단계: 결과 통합
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
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))  # 재현셩 확보용 동점 처리 (책 61쪽 참고)
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


if __name__ == '__main__':
    vocab_size = 10000
    file_path = "storybot/tiny_stories_train.txt"
    # 8개의 프로세스로 병렬 처리
    merge_rules = train_bpe(file_path, vocab_size, num_processes=8)

    with open("storybot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)