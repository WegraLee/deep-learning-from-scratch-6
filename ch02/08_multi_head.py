import torch
import torch.nn as nn
import torch.nn.functional as F

B = 2   # 배치 크기(batch_size)
C = 4   # 컨텍스트 길이(context_len)
E = 16  # 임베딩 벡터의 차원 수(embed_dim)
H = 3   # 헤드 수(n_head)
D = 8   # 헤드의 차원 수(head_dim)

# 입력 텐서
x = torch.randn(B, C, E)

# 효율적인 구현: 모든 헤드의 변환 행렬을 하나로 통합
W_q = nn.Linear(E, H*D, bias=False)
W_k = nn.Linear(E, H*D, bias=False)
W_v = nn.Linear(E, H*D, bias=False)

Q = W_q(x)  # (B, C, H*D)
K = W_k(x)  # (B, C, H*D)
V = W_v(x)  # (B, C, H*D)

# 형상 변환
Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
scores = scores / (D ** 0.5)

# 마스크 적용
mask = torch.tril(torch.ones(C, C, device=scores.device))
scores = scores.masked_fill(mask == 0, float('-inf'))

# 어텐션 가중치 계산
weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
hidden = torch.matmul(weights, V)    # (B, H, C, D)

# 형상 변환: (B, H, C, D) → (B, C, H*D)
hidden = hidden.transpose(1, 2)               # (B, C, H, D)
hidden = hidden.contiguous().view(B, C, H*D)  # (B, C, H*D)

# 출력 변환: (B, C, H*D) → (B, C, E)
W_o = nn.Linear(H*D, E, bias=False)
output = W_o(hidden)  # (B, C, E)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        # 드롭아웃 추가
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, E = x.shape                  # 배치 크기, 컨텍스트 길이, 임베딩 차원
        H, D = self.n_head, self.head_dim  # 헤드 수, 헤드의 차원 수

        # Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 각 헤드로 분할하여 재배열
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
        scores = scores / (D ** 0.5)

        # 마스크 적용
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 어텐션 가중치 계산
        weights = F.softmax(scores, dim=-1)        # (B, H, C, C)
        weights = self.attention_dropout(weights)  # weights에 드롭아웃
        hidden = torch.matmul(weights, V)          # (B, H, C, D)

        # 헤드 결합과 출력 변환
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)             # (B, C, H*D)
        output = self.W_o(hidden)                     # (B, C, E)
        output = self.output_dropout(output)          # 최종 출력에 드롭아웃 적용

        return output

# 사용 예
embed_dim = 512
n_head = 8
head_dim = 64

mha = MultiHeadAttention(embed_dim, n_head, head_dim)

# 테스트용 데이터
batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

# 실행
output = mha(x)
print(f"입력 형상: {x.shape}")       # (2, 10, 512)
print(f"출력 형상: {output.shape}")  # (2, 10, 512)