import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import numpy as np
from codebot.tokenizer import BPETokenizer


# 토크나이저 불러오기
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

# 텍스트를 토큰 ID로 변환(진행률 표시)
text = open("codebot/tiny_codes.txt").read()
ids = tokenizer.encode(text, show_progress=True)

# 넘파이 배열로 변환하여 저장
ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile("codebot/tiny_codes.bin")

print(f"토큰 ID 수: {len(ids_array)}")
print(f"처음 20개의 토큰 ID: {ids_array[:20]}")