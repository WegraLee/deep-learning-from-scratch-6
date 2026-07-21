import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from collections import defaultdict
import regex as re
from tqdm import tqdm


def pretokenize(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

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

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # 특수 토큰을 기준으로 분할
    texts = input_text.split(end_token)

    # 사전 토큰의 출현 횟수 카운트
    pretoken_counts = defaultdict(int)
    for text in tqdm(texts, desc="Pretokenizing"):  # tqdm으로 진행 상황 표시
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1

    # 사전 토큰을 ID열로 변환
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        # 각 ID열의 출현 횟수를 고려하여 ID 쌍의 출현 횟수 카운트
        pair_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            count_pairs(ids, count, pair_counts)

        # ID 쌍이 없으면 루프 종료
        if not pair_counts:
            break

        # 가장 흔한 ID 쌍 선택
        # best_pair = max(pair_counts, key=pair_counts.get)
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # ID 쌍 병합 및 ID열 갱신
        new_ids_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            new_ids = merge(ids, best_pair, new_id)  # 병합
            new_ids_counts[tuple(new_ids)] += count
        ids_counts = new_ids_counts

    return merge_rules


vocab_size = 1000  # 어휘 크기
file_path = "codebot/tiny_codes.txt"
text = open(file_path).read()
merge_rules = train_bpe(text, vocab_size)