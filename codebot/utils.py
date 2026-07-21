import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 평가 모드

    # 프롬프트 토큰화
    device = next(model.parameters()).device  # 파라미터가 있는 장치 확인
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 생성된 토큰을 저장할 변수
    generated_ids = ids.clone()

    # 토큰 생성 반복
    for _ in range(max_new_tokens):
        # 콘텍스트 길이를 초과하면 끝부분만 사용
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 다음 토큰 예측
        logits = model(ids)[:, -1, :]
        if temperature == 0:
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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')