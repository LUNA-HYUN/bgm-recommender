import os
import json
import numpy as np

# ── 1단계: BGM DB 구축 ────────────────────────────────────────
def extract_bgm_features(bgm_folder: str) -> dict:
    """
    BGM 폴더 내 오디오 파일들의 음악적 특성을 추출하여 DB로 저장
    librosa 활용
    """
    bgm_db = {}
    # TODO: librosa로 각 파일에서 tempo, spectral_centroid, energy, mfcc 추출
    return bgm_db


def save_bgm_db(bgm_db: dict, path: str = "bgm_db.json"):
    """BGM DB를 JSON 파일로 저장"""
    # TODO: 추출한 특성 벡터를 파일로 저장
    pass


def load_bgm_db(path: str = "bgm_db.json") -> dict:
    """저장된 BGM DB 불러오기"""
    # TODO: JSON 파일에서 BGM DB 로드
    pass


# ── 2단계: 텍스트 감정 분석 ──────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    KoNLPy로 한국어 형태소 분석 및 감정 키워드 추출
    """
    # TODO: KoNLPy Okt로 형태소 분석 → 형용사·동사·명사 추출
    return text


def analyze_sentiment(text: str) -> dict:
    """
    VADER로 텍스트 감정 점수 수치화
    반환: {'neg': float, 'neu': float, 'pos': float, 'compound': float}
    """
    # TODO: VADER SentimentIntensityAnalyzer로 감정 점수 산출
    return {}


def text_to_emotion_vector(text: str) -> np.ndarray:
    """
    전처리 → 감정 분석을 거쳐 장면 감정 벡터 생성
    """
    processed = preprocess_text(text)
    sentiment = analyze_sentiment(processed)
    # TODO: sentiment 딕셔너리를 numpy 벡터로 변환
    return np.array([])


# ── 3단계: BGM 매칭 ──────────────────────────────────────────
def match_bgm(emotion_vector: np.ndarray, bgm_db: dict, top_n: int = 3) -> list:
    """
    코사인 유사도로 장면 감정 벡터와 BGM DB 비교
    일치도 점수와 함께 상위 top_n개 반환
    반환: [{'filename': str, 'score': float}, ...]
    """
    # TODO: scikit-learn cosine_similarity로 유사도 계산
    # TODO: 일치도가 전반적으로 낮으면 top_n을 5개로 자동 확장
    results = []
    return results


# ── 4단계: 전체 파이프라인 ───────────────────────────────────
def recommend_bgm(scene_text: str, emotion_adjust: dict = None) -> list:
    """
    시나리오 텍스트 입력 → BGM 추천 결과 반환
    emotion_adjust: 감정 비중 수동 보정값 (슬라이더 입력)
    """
    # 1. 텍스트 감정 분석
    emotion_vector = text_to_emotion_vector(scene_text)

    # 2. 수동 보정 적용 (슬라이더 입력이 있는 경우)
    if emotion_adjust:
        # TODO: 감정 비중 수동 보정 로직
        pass

    # 3. BGM 매칭
    bgm_db = load_bgm_db()
    results = match_bgm(emotion_vector, bgm_db)

    return results


# ── 5단계: Gradio UI ─────────────────────────────────────────
def build_ui():
    """
    Gradio 웹 UI 구성
    - 시나리오 텍스트 입력창
    - 감정 비중 수동 보정 슬라이더 (긍정/부정/중립)
    - 추천 BGM 목록 + 일치도 표시
    - 미리 듣기
    """
    import gradio as gr

    # TODO: gr.Interface 또는 gr.Blocks로 UI 구성
    pass


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # BGM DB가 없으면 먼저 구축
    if not os.path.exists("bgm_db.json"):
        print("BGM DB 구축 중...")
        bgm_db = extract_bgm_features("bgm/")
        save_bgm_db(bgm_db)
        print("BGM DB 구축 완료")

    # UI 실행
    build_ui()
