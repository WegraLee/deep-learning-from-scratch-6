import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import pretokenize, count_pairs, merge

import pickle
import regex as re
from tqdm import tqdm


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
            return self.merge_rules.get(pair, float('inf'))  # 규칙 목록에 없는 ID 쌍은 우선순위를 가장 낮게 설정

        while len(ids) > 1:
            # 현재 ID열에 있는 인접한 ID 쌍들을 가져옴
            counts = count_pairs(ids)

            # 우선순위가 가장 높은 ID 쌍 찾기
            best_pair = min(counts, key=get_merge_priority)

            # 병합할 ID 쌍이 없다면 루프 종료
            if best_pair not in self.merge_rules:
                break

            # 병합
            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)

        return ids

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        texts = tqdm(texts) if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":
    tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

    file_path = "codebot/tiny_codes.txt"
    text = open(file_path).read()
    ids = tokenizer.encode(text, show_progress=True)
    print(len(ids))