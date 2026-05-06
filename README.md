# 🎮 오픈소스를 활용한 게임 시나리오 텍스트 기반 BGM 추천 시스템

경희대학교 SWCON103 디자인적 사고 프로젝트

시나리오 장면 텍스트를 입력하면 어울리는 BGM을 자동으로 추천해주는 시스템이에요.

---

## 시스템 구조

```
시나리오 텍스트 입력
        ↓
KoNLPy 형태소 분석 → deep_translator 번역 → VADER 감정 분석
        ↓
valence / arousal 감정 벡터 생성
        ↓
librosa로 추출한 BGM 특성 → DEAM 학습 모델로 감정 예측
        ↓
scikit-learn 코사인 유사도 매칭
        ↓
Gradio 웹 UI에서 추천 결과 + 미리 듣기 제공
```

---

## 사용한 오픈소스

| 라이브러리 | 역할 |
|-----------|------|
| KoNLPy | 한국어 형태소 분석 |
| deep-translator | 한국어 → 영어 번역 |
| vaderSentiment | 텍스트 감정 분석 |
| librosa | BGM 음악 특성 추출 |
| scikit-learn | 코사인 유사도 매칭 |
| joblib | 학습 모델 저장/불러오기 |
| pandas | DEAM 데이터셋 처리 |
| Gradio | 웹 UI 구현 |

---

## 설치 및 실행 방법

### 1. 레포 클론

```bash
git clone https://github.com/LUNA-HYUN/bgm-recommender.git
cd bgm-recommender
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

> KoNLPy는 Java가 필요해요.
> macOS: `brew install java ant`
> Windows: [Java 다운로드](https://www.java.com)

### 3. BGM 파일 준비

`bgm/` 폴더에 BGM 파일을 넣어주세요. (mp3, wav, ogg, flac 지원)

```
bgm-recommender/
└── bgm/
    ├── battle_theme.mp3
    ├── town_theme.mp3
    └── ...
```

> BGM 파일은 저작권 문제로 포함되어 있지 않아요.
> 로열티 프리 음원을 권장해요: [opengameart.org](https://opengameart.org)

### 4. 실행

```bash
python bgm_recommender.py
```

실행하면 자동으로 BGM DB를 구축하고 웹 UI가 열려요.
브라우저에서 `http://127.0.0.1:7860` 으로 접속하세요.

---

## 감정 예측 모델

`emotion_model.pkl` 이 포함되어 있어요. DEAM 데이터셋으로 학습된 RandomForest 모델로, BGM 음악 특성에서 valence/arousal을 예측해요.

모델을 직접 학습하려면 DEAM 데이터셋이 필요해요.

```bash
# DEAM 다운로드: https://cvml.unige.ch/databases/DEAM/
# deam/ 폴더에 압축 해제 후

python train_model.py
```

---

## 주요 기능

- 시나리오 텍스트 → 자동 감정 분석 (valence/arousal)
- 긴 텍스트 입력 시 감정이 강한 핵심 문장 자동 추출
- 감정 비중 수동 보정 슬라이더
- 일치도 낮으면 추천 수 자동 확장 (3개 → 5개)
- 추천 BGM 미리 듣기
- 더보기 버튼으로 전체 추천 목록 확인

---

## 한계점

- valence/arousal 2차원 감정 모델로는 "긴장감 있는 전투"와 "활기찬 일상"을 구분하기 어려움
- DEAM(일반 음악)과 게임 BGM 간의 도메인 불일치
- VADER가 영어 기반이라 한국어 감정 분석 정확도에 한계 있음
