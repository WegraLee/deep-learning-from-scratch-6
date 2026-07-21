import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# 설정
device = get_device()
model_path = 'storybot/model_pretrain.pt'
tokenizer_path = 'storybot/merge_rules.pkl'

# 생성 설정
# prompt = "Once upon a time"  # 생성을 시작할 프롬프트
prompt = "<|endoftext|>"
max_new_tokens = 300  # 생성할 토큰 수의 상한
temperature = 1.0     # 온도 파라미터(높을수록 무작위성이 커짐)
num_samples = 3       # 생성할 샘플 수

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# 텍스트 생성
for i in range(num_samples):
    print(f"--- 샘플 {i+1} ---")
    story = generate(
        model, tokenizer, prompt, max_new_tokens, temperature
    )
    print(story)