import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
import torch
import torch.nn.functional as F
from openai import OpenAI
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# 설정
# ==========================================
client = OpenAI(api_key="your_api_key_here")
# ==========================================
device = get_device()
tokenizer_path = 'storybot/merge_rules.pkl'
tokenizer = BPETokenizer.load_from(tokenizer_path)

# 비교할 모델
model_paths = {
    'pretrain': 'storybot/model_pretrain.pt',
    'dpo': 'storybot/model_dpo.pt',
}

# 평가 설정
prompt = "Once upon a time"
num_comparisons = 100  # 비교 횟수
max_new_tokens = 150
temperature = 1.0


def compare_stories(client, story_a, story_b):
    """두 스토리를 비교하여 어느 쪽이 더 해피엔딩에 가까운지 판정"""

    evaluation_prompt = f"""다음의 두 어린이용 스토리를 비교하여, 어느 쪽이 더 해피엔딩인지 판단해주세요.

【Story A】
{story_a}

【Story B】
{story_b}

어느 쪽의 결말이 더 밝고 행복한지, 또는 내용이 더 희망적인지 판단해주세요.
JSON형식으로 답변: {{"winner": "A" or "B" or "tie", "reason": "간단한 이유"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=150,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    return json.loads(text)

# 모델 불러오기
model_pretrain = GPT.load_from(model_paths['pretrain'], device=device)
model_dpo = GPT.load_from(model_paths['dpo'], device=device)

# 결과 기록
results = []
wins = {"pretrain": 0, "dpo": 0, "tie": 0}

for i in range(num_comparisons):
    print(f"\n{'='*60}")
    print(f"Comparison {i+1}/{num_comparisons}")
    print('='*60)

    # 두 모델로 스토리 생성
    story_pretrain = generate(model_pretrain, tokenizer, prompt, max_new_tokens, temperature)
    story_dpo = generate(model_dpo, tokenizer, prompt, max_new_tokens, temperature)

    print(f"\n[Pretrain]: {story_pretrain[:100]}...")
    print(f"\n[DPO]: {story_dpo[:100]}...")

    # 위치 편향을 방지하기 위해 순서를 무작위로 바꿈
    import random
    if random.random() < 0.5:
        story_a, story_b = story_pretrain, story_dpo
        mapping = {"A": "pretrain", "B": "dpo"}
    else:
        story_a, story_b = story_dpo, story_pretrain
        mapping = {"A": "dpo", "B": "pretrain"}

    # LLM-as-a-Judge로 비교
    judgment = compare_stories(client, story_a, story_b)

    winner_label = judgment["winner"]
    if winner_label == "tie":
        winner = "tie"
    else:
        winner = mapping[winner_label]

    wins[winner] += 1

    print(f"\n🏆 승자: {winner}")
    print(f"   이유: {judgment['reason']}")

    results.append({
        "story_pretrain": story_pretrain,
        "story_dpo": story_dpo,
        "winner": winner,
        "reason": judgment["reason"]
    })

# 요약 출력
print("\n" + "="*60)
print("📊 일대일 비교 결과")
print("="*60)

total = num_comparisons
print(f"\n  사전 학습 모델 승리: {wins['pretrain']:3d} ({wins['pretrain']/total*100:5.1f}%)")
print(f"  DPO 모델 승리:      {wins['dpo']:3d} ({wins['dpo']/total*100:5.1f}%)")
print(f"  무승부:          {wins['tie']:3d} ({wins['tie']/total*100:5.1f}%)")

# 승률(무승부 제외)
if wins['pretrain'] + wins['dpo'] > 0:
    dpo_winrate = wins['dpo'] / (wins['pretrain'] + wins['dpo']) * 100
    print(f"\n  DPO 모델 승률(무승부 제외): {dpo_winrate:.1f}%")