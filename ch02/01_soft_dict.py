import numpy as np

# 딕셔너리 사용 예
d = {
    'apple': 100,
    'banana': 200,
    'cherry': 300,
    'durian': 400
}

# 키를 지정해 밸류 가져오기
query = 'banana'
print(d[query])  # 200


# 키: (액션성, 드라마성, 코미디성) 각각 0~10으로 표현
# 밸류: 사용자가 매긴 평점(0~100점)
movie_preferences = {
    (8, 2, 3): 85,   # 액션 중심 영화
    (3, 9, 1): 70,   # 드라마 중심 영화
    (1, 2, 9): 60,   # 코미디 중심 영화
    (5, 5, 5): 75,   # 균형 잡힌 영화
    (7, 6, 2): 80,   # 액션 드라마
    (2, 7, 6): 65,   # 코미디 드라마
    (9, 1, 1): 90,   # 순수 액션 영화
}

# 새로 평가할 영화
new_movie = (6, 4, 5)

def soft_dictionary(query, dictionary):
    # 유사도
    similarity = []
    for key in dictionary:
        s = np.dot(query, key)
        similarity.append(s)

    # 소프트맥스
    exp_similarity = np.exp(similarity)
    weights = exp_similarity / np.sum(exp_similarity)

    # 가중합
    result = 0
    for weight, value in zip(weights, dictionary.values()):
        result += weight * value

    return result, weights


predicted_rating, weights = soft_dictionary(new_movie, movie_preferences)

print(f"새로운 영화 {new_movie}의 예측 평점: {predicted_rating:.2f}점")
print("\n각 영화의 가중치:")
for key, weight in zip(movie_preferences.keys(), weights):
    print(f"영화 {key}: {weight*100:.2f}%")