import torch
import torch.nn.functional as F

# 키(영화의 장르 특성)
K = torch.tensor([
    [8, 2, 3],  # 액션 중심 영화
    [3, 9, 1],  # 드라마 중심 영화
    [1, 2, 9],  # 코미디 중심 영화
    [5, 5, 5],  # 균형 잡힌 영화
    [7, 6, 2],  # 액션 드라마
    [2, 7, 6],  # 코미디 드라마
    [9, 1, 1],  # 순수 액션 영화
], dtype=torch.float32)

# 밸류(사용자 평점)
V = torch.tensor([
    [85],
    [70],
    [60],
    [75],
    [80],
    [65],
    [90]
], dtype=torch.float32)

# 새로 평가할 영화의 장르 특성(여러 쿼리)
Q = torch.tensor([
    [6, 4, 5],  # 균형 잡힌 액션 성향의 영화
    [2, 8, 3],  # 드라마 중심 영화
    [4, 3, 7],  # 코미디 성향의 영화
], dtype=torch.float32)

def attention(Q, K, V):
    similarity = torch.matmul(Q, K.t())     # QK^T 계산
    weights = F.softmax(similarity, dim=1)  # 소프트맥스 함수
    output = torch.matmul(weights, V)       # 가중합
    return output, weights

predicted_ratings, weights = attention(Q, K, V)

# 결과 출력
for movie, rating in zip(Q, predicted_ratings):
    print(f"영화 {movie.numpy()}의 예측 평점: {rating.item():.2f}")