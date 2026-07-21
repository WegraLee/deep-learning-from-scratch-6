import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from webbot.model import GPT
from storybot.tokenizer import BPETokenizer
from webbot.utils import generate, get_device

# 설정
device = get_device()
model_path = 'webbot/model_pretrain.pt'
tokenizer_path = 'webbot/merge_rules.pkl'
max_new_tokens = 100
temperature = 0.5

# 테스트용 프롬프트
prompts = [
    "In 1991, Linus Torvalds created",
    "Monday, Tuesday, Wednesday,",
    "Python was created by",
    "Machine learning is defined as",
    "The capital of Japan is",
]

# 모델과 토크나이저 불러오기
print("모델과 토크나이저 불러오는 중...")
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)
print(f"불러오기 완료!\n")

# 각 프롬프트로 텍스트 생성
for i, prompt in enumerate(prompts, 1):
    print(f"{'=' * 70}")
    print(f"프롬프트 {i}: {prompt}")
    print(f"{'=' * 70}")

    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    print(response)
    print()
