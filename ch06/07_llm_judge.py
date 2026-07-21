import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
import statistics
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

# 평가할 모델 경로(이터레이션별로 저장한 모델)
model_paths = {
    500: 'storybot/model_iter_500.pt',
    5000: 'storybot/model_iter_5000.pt',
    40000: 'storybot/model_pretrain.pt',
}

# 생성 설정
prompt = "<|endoftext|>"
max_new_tokens = 200
temperature = 1.0
num_samples = 10  # 각 모델에서 생성할 샘플 수

def evaluate_story(client, story):
    """LLM-as-a-Judge로 스토리 평가"""

    evaluation_prompt = f"""다음 어린이용 스토리를 두 가지 관점에서 1~5점으로 평가해주세요.

스토리:
{story}

평가 관점:
1. Coherence(일관성): 논리적으로 이어지는가, 이야기로서 앞뒤가 맞는가
2. Grammar(문법): 문법적으로 올바른 영어인가

다음 JSON 형식으로 답변해주세요.
{{
    "coherence": <1-5 범위의 정수>,
    "grammar": <1-5 범위의 정수>,
    "comment": "<평가에 대한 간단한 이유>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=300,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    print("===== 출력 ====")
    print(text)

    # response_format을 사용하면 파싱 처리가 간단해짐
    return json.loads(text)

results = {}
for iteration, model_path in model_paths.items():
    print(f"\n{'='*50}")
    print(f"Iteration {iteration}")
    print('='*50)

    model = GPT.load_from(model_path, device=device)
    iteration_results = []

    for i in range(num_samples):
        print(f"\n--- 샘플 {i+1} ---")

        # 스토리 생성
        story = generate(model, tokenizer, prompt, max_new_tokens, temperature)
        print(f"Story: {story[:200]}...")

        # LLM-as-a-Judge로 평가
        scores = evaluate_story(client, story)
        print(f"Scores: {scores}")

        iteration_results.append({
            "story": story,
            "scores": scores
        })

    results[iteration] = iteration_results

# 요약 출력
print("\n" + "="*50)
print("요약")
print("="*50)

for iteration in model_paths.keys():
    scores_list = [r["scores"] for r in results[iteration]]

    print(f"\n이터레이션 {iteration}:")
    for key in ["일관성", "문법"]:
        values = [s[key] for s in scores_list]
        avg = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {key}: {avg:.2f} ± {std:.2f}")