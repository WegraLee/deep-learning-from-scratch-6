from collections import defaultdict

def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
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


class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # ID와 바이트열의 대응표 생성(0~255 등록)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        # 병합된 토큰은 원래 토큰의 바이트열을 연결해 생성
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        # 어휘 크기 설정
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        # 학습할 때와 같은 순서로 병합 규칙 적용
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids

    def decode(self, ids):
        # 각 토큰 ID를 대응하는 바이트열로 변환
        byte_list = [self.id_to_bytes[i] for i in ids]

        # 모든 바이트열을 연결
        combined_bytes = b"".join(byte_list)

        # 바이트열을 UTF-8 텍스트로 변환
        text = combined_bytes.decode("utf-8", errors="replace")
        return text

# 학습된 병합 규칙
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

# 토크나이저 생성
tokenizer = BPETokenizer(merge_rules)

# 텍스트 인코딩
text = "Hello월드😁"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)  # [259, 108, 108, 111, 236, 155, 148, 235, 147, 156, 240, 159, 152, 129]
print(decoded)  # Hello월드😁