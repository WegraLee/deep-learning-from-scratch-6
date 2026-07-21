import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim % 2 == 0  # key_dim은 짝수
        half = key_dim // 2

        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ( (2.0 * half_ids) / key_dim ))  # (half,)

        positions = torch.arange(max_context_len)        # (max_context_len,)
        angles = positions[:, None] * inv_freq[None, :]  # (max_context_len, half)

        cos = torch.cos(angles)  # (max_context_len, half)
        sin = torch.sin(angles)  # (max_context_len, half)

        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, x):
        batch_size, num_head, context_len, key_dim = x.shape

        # 입력의 타입을 저장해두고, float32로 계산
        input_dtype = x.dtype
        x = x.float()

        # 캐시에서 cos와 sin 값 불러오기
        cos = self.cos_cache[:context_len]
        sin = self.sin_cache[:context_len]

        # 짝수·홀수 인덱스로 분할
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        # 회전 적용
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        # 짝수·홀수 인덱스를 원래 순서로 복원
        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (batch_size, num_head, context_len, key_dim/2, 2)
        out = out.reshape(batch_size, num_head, context_len, key_dim)

        return out.to(input_dtype)  # 원래의 타입으로 복원

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, rope=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.rope = rope

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        # RoPE 적용
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (D ** 0.5)

        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        return output

# 하이퍼파라미터
embed_dim = 512
n_head = 8
head_dim = 64
theta = 10000
max_context_len = 1024

# 초기화
rope = RoPE(theta, head_dim, max_context_len)
mha = MultiHeadAttention(embed_dim, n_head, head_dim, rope=rope)

# 테스트용 데이터
batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

# 순전파
output = mha(x)
print(output.shape)  # (2, 10, 512)