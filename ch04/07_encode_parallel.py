import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import pretokenize, count_pairs, merge, find_chunk_boundaries

import os
import pickle
from multiprocessing import Pool
import shutil
import regex as re
from tqdm import tqdm
import numpy as np


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
            return self.merge_rules.get(pair, float('inf'))  # 규치 목록에 없는 ID 쌍은 우선순위를 가장 낮게 설정

        while len(ids) > 1:
            # 현재 ID열에 있는 인접한 ID 쌍들을 가져옴
            counts = count_pairs(ids)

            # 우선순위가 가장 높은 ID 쌍 찾기
            best_pair = min(counts, key=get_merge_priority)

            # 병합할 수 있는지 확인
            if best_pair not in self.merge_rules:
                break

            # 병합
            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)

        return ids

    def _encode_chunk(self, args):
        """개별 청크를 처리하고 캐시 파일에 저장"""
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
                                    num_processes=8, num_chunks=64,
                                   cache_dir="bpe_cache"):

        # 캐시 디렉터리 준비
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # 청크들을 병렬로 인코딩
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
            dtype = np.uint16  # 각 토큰 ID를 uint16 타입으로 저장 
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

            # 캐시 파일의 데이터를 메모리 맵 파일에 기록
            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            arr.flush()  # 저장 장치에 반영
            del arr      # 메모리 맵 객체 삭제

        finally:
            # 캐시 디렉터리 삭제
            shutil.rmtree(cache_dir)

        return total_tokens

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

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

    tokenizer.encode_file(
        "storybot/tiny_stories_train.txt",
        "storybot/tiny_stories_train.bin", num_processes=8)

    tokenizer.encode_file(
        "storybot/tiny_stories_valid.txt",
        "storybot/tiny_stories_valid.bin", num_processes=8)