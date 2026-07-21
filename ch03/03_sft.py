import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device

# 설정
device = get_device()
data_path = 'codebot/tiny_codes_sft.json'
tokenizer_path = 'codebot/merge_rules.pkl'
pretrain_model_path = 'codebot/model_pretrain.pt'
sft_model_save_path = 'codebot/model_sft.pt'

# 하이퍼파라미터
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 500

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            ids, labels = self._create_sample(item['instruction'], item['response'])
            self.samples.append((ids, labels))

    # 샘플 생성
    def _create_sample(self, instruction, response):
        # 프롬프트(지시)와 응답 포맷팅
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"

        # 토큰화
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        # 입력 시퀀스와 레이블 생성(프롬프트 부분은 -100으로 마스킹)
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 언어 모델용으로 시프트(입력과 정답이 한 칸씩 어긋나게)
        ids = ids[:-1]
        labels = labels[1:]

        # context_len에 맞춰 패딩 혹은 잘라내기
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len  # 패딩 ID로 0 사용
            labels = labels + [-100] * pad_len
        elif pad_len < 0:
            ids = ids[:self.context_len]
            labels = labels[:self.context_len]

        return ids, labels

    # DataLoader용 메서드
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.long)


# 토크나이저와 데이터셋 준비
tokenizer = BPETokenizer.load_from(tokenizer_path)
dataset = SFTDataset(data_path, tokenizer, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델과 옵티마이저
model = GPT.load_from(pretrain_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 학습 루프
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_y.view(-1),
        ignore_index=-100  # 레이블이 -100인 위치를 손실 계산에서 제외
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# 결과 저장
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_sft.png')

# 모델 저장
model.save(sft_model_save_path)