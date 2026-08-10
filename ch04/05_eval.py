import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import BPETokenizer


tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

# 4.5.1 학습된 토큰 확인
print("가장 먼저 학습된 10개:")
for token_id in range(256, 266):
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8")
    print(f"  ID {token_id}: '{text}'")

print("\n마지막에 학습된 10개:")
for token_id in range(9990, 10000):
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8")
    print(f"  ID {token_id}: '{text}'")


# 4.5.2 압축 효율 측정
sample_text = open("storybot/tiny_stories_train.txt").read()[:10000]  # 처음 10,000자
byte_count = len(sample_text.encode("utf-8"))
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print(f"\n바이트 수: {byte_count:,}")
print(f"토큰 수: {ids_count:,}")
print(f"압축률: {compression_ratio:.2f}배 (평균 {compression_ratio:.2f} 바이트/토큰)")


# 4.5.3 CodeBot 토크나이저와 비교
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print("\n=== CodeBot 토크나이저의 압축 효율 ===")
print(f"바이트 수: {byte_count:,}")
print(f"토큰 수: {ids_count:,}")
print(f"압축률: {compression_ratio:.2f}배 (평균 {compression_ratio:.2f} 바이트/토큰)")