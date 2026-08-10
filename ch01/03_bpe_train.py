from collections import defaultdict

def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# 사용 예
ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
print(counts)  # {(1, 2): 2, (2, 3): 1, (3, 1): 1}

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

# 사용 예
ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1, 2), 4)
print(merged)  # [4, 3, 4]

def train_bpe(text, vocab_size):
    # 텍스트를 0~255의 ID열로 변환
    ids = list(text.encode("utf-8"))

    # 병합 횟수 결정
    num_merges = vocab_size - 256  # 256은 초기 어휘 크기
    merge_rules = {}

    for step in range(num_merges):
        # 인접한 ID 쌍 집계
        counts = count_pairs(ids)

        # 인접한 ID 쌍이 존재하지 않으면 루프 종료
        if not counts:
            break

        # 가장 빈번한 쌍 선택
        best_pair = max(counts, key=counts.get)
        # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))  # 재현셩 확보용 동점 처리 (책 61쪽 참고)

        # 새로운 토큰 ID 할당
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # 병합
        ids = merge(ids, best_pair, new_id)

    return merge_rules

# 사용 예
text = "Hello world! This is BPE training."
merge_rules = train_bpe(text, vocab_size=260)  # BPE 학습
print(merge_rules)  # {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}