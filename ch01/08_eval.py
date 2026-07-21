import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from codebot.tokenizer import BPETokenizer


tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

print("처음에 학습된 10개:")
for token_id in range(256, 266):
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8")
    print(f"  ID {token_id}: '{text}'")

print("\n마지막에 학습된 10개:")
for token_id in range(990, 1000):
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8")
    print(f"  ID {token_id}: '{text}'")


# 압축률 측정
sample_text = open("codebot/tiny_codes.txt").read()[:10000]  # 처음 10,000자

byte_count = len(sample_text.encode("utf-8"))
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print("\n=== 압축 효율 ===")
print(f"바이트 수: {byte_count:,}")
print(f"토큰 수: {ids_count:,}")
print(f"압축률: {compression_ratio:.2f}배(평균 {compression_ratio:.2f} 바이트/토큰)")


# ==== 다음은 GPT 계열 모델의 인코딩 및 압축률 비교(tiktoken 사용) ====
"""
import tiktoken

text = open("codebot/tiny_codes.txt").read()[:10000]
byte_count = len(text.encode("utf-8"))

for name, encoding_name in [('GPT-2', 'gpt2'), ('cl100k_base', 'cl100k_base')]:
    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text, allowed_special={'<|endoftext|>'}))
    ratio = byte_count / token_count
    print(f"{name}: 어휘 크기 {encoding.n_vocab:,}, 압축률 {ratio:.2f}배")
"""