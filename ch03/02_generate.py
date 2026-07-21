import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import torch
import torch.nn.functional as F
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device


# 설정
device = get_device()
model_path = 'codebot/model_pretrain.pt'
tokenizer_path = 'codebot/merge_rules.pkl'

# 생성 설정
prompt = "def"        # 생성을 시작할 프롬프트
max_new_tokens = 200  # 생성할 토큰 수의 상한
temperature = 1.0     # 온도 파라미터(높을수록 무작위성 커짐)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 평가 모드

    # 프롬프트 토큰화
    device = next(model.parameters()).device  # 모델 파라미터의 디바이스를 가져옴
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 생성된 전체 토큰을 보관하는 변수
    generated_ids = ids.clone()

    # 토큰 생성 루프
    for _ in range(max_new_tokens):
        # 최대 컨텍스트 길이를 초과하면 오래된 토큰을 잘라냄
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 마지막 위치의 로짓을 가져옴(다음 토큰 예측)
        logits = model(ids)[:, -1, :]
        if temperature == 0:  # 온도가 0이면 최댓값의 인덱스를 가져옴
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 종료 토큰이 생성되면 중단
        if next_id.item() == tokenizer.end_token_id:
            break

        # 생성한 토큰 추가
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # 디코딩하여 반환
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# 텍스트 생성
for i in range(5):
    print(f"--- 샘플 {i+1} ---")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(generated_text)
    print()