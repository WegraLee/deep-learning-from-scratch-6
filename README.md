『밑바닥부터 시작하는 딥러닝 ❻』 <br>: 토크나이저부터 추론 모델까지 직접 만드는 LLM
=============================

<a href="https://product.kyobobook.co.kr/detail/S000220996795"><img src="https://github.com/WegraLee/deep-learning-from-scratch-6/blob/main/cover_3d.png?raw=true" width="180" align=right></a>


**처음부터 끝까지 구현하며 배우는 LLM의 원리**

챗GPT 같은 대규모 언어 모델(LLM)이 일상에 깊이 들어온 지금, 그 작동 원리를 직접 구현해 본 사람은 많지 않습니다. 『밑바닥부터 시작하는 딥러닝』 여섯 번째 편은 토크나이저, 모델, 학습이라는 LLM의 핵심 세 단계를 완성된 라이브러리에 기대지 않고 밑바닥부터 구현합니다. CodeBot, StoryBot, WebBot 세 프로젝트를 거치며 BPE 토크나이저, 트랜스포머 어텐션, 사전 학습부터 SFT·GRPO·DPO 강화 학습까지 LLM 개발의 전 과정을 코드로 직접 재현합니다. 파이썬, 파이토치 기초와 딥러닝에 대한 기본 이해가 있다면, 이 책과 함께 나만의 LLM을 밑바닥부터 키워보세요.


[알려진 오류(정오표)](https://docs.google.com/document/d/1OmaYaKZPlcHpygzp61qmJUTe5tKRceMaMJRxx9H-k9s)


## 파일 구성

|폴더명 |설명                             |
|:--        |:--                              |
|`ch01`〜`ch09`|각 장에서 사용하는 코드|
|`codebot`   |CodeBot에서 사용하는 코드와 데이터 |
|`storybot`   |StoryBot에서 사용하는 코드와 데이터 |
|`webbot`   |WebBot에서 사용하는 코드와 데이터 |
<!-- |`notebooks`   |1章〜6章までのコード（Jupyter Notebook形式）| -->

## 파이썬과 외부 라이브러리

소스 코드를 실행하려면 다음 라이브러리가 필요합니다.

* NumPy
* Matplotlib
* PyTorch 2.x
* tqdm

※ 파이썬은 3.10 이상을 사용합니다.


## 환경 구축

### uv를 사용하는 경우

[uv](https://docs.astral.sh/uv/)는 빠른 파이썬 패키지 관리자입니다.`uv.lock`을 사용하면 운영체제와 관계없이 동일한 환경을 재현할 수 있습니다.
```bash
# uv 설치(설치되어 있지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Mac/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

실행할 때는 `uv run`을 사용합니다. 필요한 환경도 자동으로 구축됩니다.
```bash
uv run python ch01/01_char_tokenizer.py
```

### pip를 사용하는 경우
```bash
pip install -r requirements.txt
python ch01/01_char_tokenizer.py
```


## 실행 방법

각 장의 폴더로 이동하여 실행하거나, 상위 폴더에서 실행합니다.
```bash
# 각 장의 폴더에서 실행
$ cd ch01
$ python 01_char_tokenizer.py

# 상위 폴더에서 실행
$ python ch02/10_gpt.py
```

uv를 사용하는 경우에는 `python`을 `uv run python`으로 바꾸어 실행하세요.

## 라이선스

이 리포지터리의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)로 제공됩니다. 상업적·비상업적 용도와 관계없이 자유롭게 이용할 수 있습니다.