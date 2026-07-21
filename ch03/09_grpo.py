import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device


# 데이터셋
class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, prompts, responses, device):
        all_ids = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(response)

            ids = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)

        # 패딩
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.float, device=device)

        return ids, mask


# 보상 함수
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])  # 마지막 정수 가져오기
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


# 그룹 생성
def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            response = full_text[len(prompt):]
            responses.append(response)

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, torch.stack(all_advantages)

# 손실 함수
def compute_probs(model, ids):
    logits = model(ids)  # (B, C, V)
    probs = F.softmax(logits[:, :-1, :], dim=-1)  # (B, C-1, V)
    labels = ids[:, 1:]  # (B, C-1)

    token_probs = torch.gather(
        probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, C-1)

    return token_probs

def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    # 현재 모델의 토큰별 확률
    probs = compute_probs(model, ids)
    # 옛 모델의 토큰별 확률
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    # 토큰별 확률비(0으로 나눠지는 일을 막기 위해 작은 값을 더함)
    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]  # 마스크도 한 칸 시프트
    token_objective = torch.min(unclipped, clipped) * mask

    # 샘플 수(batch_size × group_size)로 정규화
    n_samples = ids.size(0)  # batch_size × group_size
    return -token_objective.sum() / n_samples


# 설정
device = get_device()
tokenizer_path = 'codebot/merge_rules.pkl'
sft_model_path = 'codebot/model_sft.pt'
grpo_model_save_path = 'codebot/model_grpo.pt'

# 하이퍼파라미터
learning_rate = 7e-6
max_iters = 500
n_update_per_generation = 2  # 같은 생성 데이터로 갱신하는 횟수
eval_interval = 10
epsilon = 0.2   # 클리핑 범위
group_size = 8  # 그룹 크기
batch_size = 32

# 초기화
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(sft_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

old_model = GPT.load_from(sft_model_path, device=device)  # 옛 모델
old_model.eval()

dataset = GRPODataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = cycle(dataloader)

# 학습 루프
accuracies = []
current_accuracy = 0.0
pbar = tqdm(range(max_iters))

for i in pbar:
    # 배치 데이터 가져오기
    prompts, gts = next(data_iter)

    # 옛 모델(old_model) 갱신
    old_model.load_state_dict(model.state_dict())

    # 옛 모델로 여러 샘플을 생성하고, 보상과 어드밴티지 계산
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size
    )

    # 배치 데이터 만들기
    ids, mask = dataset.get_batch(all_prompts, all_responses, device)
    all_advantages = all_advantages.to(device)

    # 생성 데이터로 여러 번 갱신
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 기울기 클리핑
        optimizer.step()

    # 주기적으로 평가
    if i % eval_interval == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for prompt, gt in dataset.data:
                response = generate(model, tokenizer, prompt, temperature=0)
                reward = calculate_reward(gt, response)
                correct += reward > 0
                total += 1
        model.train()
        current_accuracy = correct / total * 100
        accuracies.append(current_accuracy)

    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_accuracy:.1f}%'})

# 학습된 모델 저장
model.save(grpo_model_save_path)

plt.figure()
steps = list(range(0, len(accuracies) * eval_interval, eval_interval))
plt.plot(steps, accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('GRPO Training')
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_grpo.png")