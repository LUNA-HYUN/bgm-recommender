import os
import json
import numpy as np

# ── 1단계: BGM DB 구축 ────────────────────────────────────────
def extract_bgm_features(bgm_folder: str) -> dict:
    """
    BGM 폴더 내 오디오 파일들의 음악적 특성을 추출하여 DB로 저장
    librosa 활용
    """
    import librosa

    bgm_db = {}
    supported = ('.mp3', '.wav', '.ogg', '.flac')

    for filename in os.listdir(bgm_folder):
        if not filename.lower().endswith(supported):
            continue

        filepath = os.path.join(bgm_folder, filename)
        print(f"추출 중: {filename}")

        try:
            # 오디오 로드 (모노, 22050Hz)
            y, sr = librosa.load(filepath, sr=22050, mono=True)

            # tempo (빠르기)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo)[0])

            # spectral centroid 평균 (음색 밝기)
            spectral_centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr)
            ))

            # RMS energy 평균 (음량/에너지)
            rms = float(np.mean(librosa.feature.rms(y=y)))

            # zero crossing rate 평균 (타악기성/노이즈성)
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

            # spectral contrast 평균 (스펙트럼 피크와 골의 차이)
            spectral_contrast = float(np.mean(
                librosa.feature.spectral_contrast(y=y, sr=sr)
            ))

            # MFCC 평균 (음색 특성, 13개 계수)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1).tolist()

            bgm_db[filename] = {
                'tempo': tempo,
                'spectral_centroid': spectral_centroid,
                'rms': rms,
                'zcr': zcr,
                'spectral_contrast': spectral_contrast,
                'mfcc': mfcc_mean,
            }
            print(f"  완료: tempo={tempo:.1f}, rms={rms:.4f}, zcr={zcr:.4f}, contrast={spectral_contrast:.2f}")

        except Exception as e:
            print(f"  오류 ({filename}): {e}")

    return bgm_db


def save_bgm_db(bgm_db: dict, path: str = "bgm_db.json"):
    """BGM DB를 JSON 파일로 저장"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(bgm_db, f, ensure_ascii=False, indent=2)
    print(f"\nBGM DB 저장 완료: {path} ({len(bgm_db)}개 트랙)")


def load_bgm_db(path: str = "bgm_db.json") -> dict:
    """저장된 BGM DB 불러오기"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── 2단계: 텍스트 감정 분석 ──────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    KoNLPy Okt로 한국어 형태소 분석
    형용사·동사·명사만 추출 후 영어로 번역하여 반환
    """
    from konlpy.tag import Okt
    from deep_translator import GoogleTranslator

    okt = Okt()

    # 형용사(Adjective), 동사(Verb), 명사(Noun)만 추출
    target_pos = {'Adjective', 'Verb', 'Noun'}
    tokens = [
        word for word, pos in okt.pos(text, stem=True)
        if pos in target_pos
    ]
    korean = ' '.join(tokens)
    print(f"  형태소 추출: {korean}")

    # 영어로 번역
    translated = GoogleTranslator(source='ko', target='en').translate(korean)
    print(f"  영어 번역: {translated}")
    return translated


def analyze_sentiment(text: str) -> dict:
    """
    VADER로 텍스트 감정 점수 수치화
    반환: {'neg': float, 'neu': float, 'pos': float, 'compound': float}
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    print(f"  감정 점수: {scores}")
    return scores


def text_to_emotion_vector(text: str) -> np.ndarray:
    """
    전처리 → 감정 분석을 거쳐 장면 감정 벡터 생성
    벡터 구성: [neg, neu, pos, compound]
    """
    processed = preprocess_text(text)
    sentiment = analyze_sentiment(processed)

    vector = np.array([
        sentiment['neg'],
        sentiment['neu'],
        sentiment['pos'],
        sentiment['compound'],
    ])
    print(f"  감정 벡터: {vector}")
    return vector


# ── 3단계: BGM 매칭 ──────────────────────────────────────────
def bgm_features_to_emotion_vector(features: dict) -> np.ndarray:
    """
    BGM 음악 특성을 감정 벡터 [neg, neu, pos, compound]로 변환

    매핑 규칙:
    - spectral_centroid 높음 → 밝고 긍정적 (pos)
    - spectral_centroid 낮음 → 어둡고 부정적 (neg)
    - zcr 높음 + rms 높음    → 타악기적/전투적 (neg)
    - spectral_contrast 높음 → 강렬하고 긴장감 있음 (neg)
    - tempo 낮음 + rms 낮음  → 차분/평화로움 (pos)
    """
    tempo = features['tempo']
    rms = features['rms']
    spectral_centroid = features['spectral_centroid']
    zcr = features.get('zcr', 0.05)
    spectral_contrast = features.get('spectral_contrast', 20.0)

    # 정규화 기준값 (경험적 범위)
    TEMPO_MIN, TEMPO_MAX = 60.0, 200.0
    RMS_MIN, RMS_MAX = 0.01, 0.3
    SC_MIN, SC_MAX = 200.0, 4000.0
    ZCR_MIN, ZCR_MAX = 0.01, 0.2
    CONTRAST_MIN, CONTRAST_MAX = 5.0, 40.0

    # 0~1 범위로 정규화
    tempo_norm = np.clip((tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN), 0, 1)
    rms_norm = np.clip((rms - RMS_MIN) / (RMS_MAX - RMS_MIN), 0, 1)
    sc_norm = np.clip((spectral_centroid - SC_MIN) / (SC_MAX - SC_MIN), 0, 1)
    zcr_norm = np.clip((zcr - ZCR_MIN) / (ZCR_MAX - ZCR_MIN), 0, 1)
    contrast_norm = np.clip((spectral_contrast - CONTRAST_MIN) / (CONTRAST_MAX - CONTRAST_MIN), 0, 1)

    # 긍정 점수: 밝은 음색 + 조용하고 차분한 느낌
    pos = sc_norm * 0.5 + (1 - rms_norm) * 0.3 + (1 - zcr_norm) * 0.2

    # 부정 점수: 어두운 음색 + 타악기적 + 강렬한 대비 + 빠른 템포
    neg = (1 - sc_norm) * 0.35 + zcr_norm * 0.25 + contrast_norm * 0.25 + tempo_norm * 0.15

    # 중립 점수
    neu = 1.0 - (pos + neg) / 2

    # compound: -1 ~ 1 범위
    compound = np.clip(pos - neg, -1.0, 1.0)

    return np.array([neg, neu, pos, compound])


def match_bgm(emotion_vector: np.ndarray, bgm_db: dict, top_n: int = 3) -> list:
    """
    코사인 유사도로 장면 감정 벡터와 BGM DB 비교
    일치도 점수와 함께 상위 top_n개 반환
    반환: [{'filename': str, 'score': float}, ...]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    results = []

    for filename, features in bgm_db.items():
        bgm_vector = bgm_features_to_emotion_vector(features)

        # 코사인 유사도 계산 (-1~1 → 0~1로 정규화)
        score = cosine_similarity(
            emotion_vector.reshape(1, -1),
            bgm_vector.reshape(1, -1)
        )[0][0]
        score = float((score + 1) / 2)  # 0~1 범위로 변환

        results.append({'filename': filename, 'score': score})

    # 일치도 기준 내림차순 정렬
    results.sort(key=lambda x: x['score'], reverse=True)

    # 일치도가 전반적으로 낮으면 top_n을 5개로 자동 확장
    top_score = results[0]['score'] if results else 0
    if top_score < 0.6:
        top_n = min(5, len(results))
        print(f"  일치도가 낮아 추천 수를 {top_n}개로 확장")

    return results[:top_n]


# ── 4단계: 전체 파이프라인 ───────────────────────────────────
def recommend_bgm(scene_text: str, emotion_adjust: dict = None) -> list:
    """
    시나리오 텍스트 입력 → BGM 추천 결과 반환
    emotion_adjust: 감정 비중 수동 보정값 (슬라이더 입력)
    """
    emotion_vector = text_to_emotion_vector(scene_text)

    if emotion_adjust:
        # 수동 보정값으로 감정 벡터 덮어쓰기
        compound = np.clip(emotion_adjust['pos'] - emotion_adjust['neg'], -1.0, 1.0)
        emotion_vector = np.array([
            emotion_adjust['neg'],
            emotion_adjust['neu'],
            emotion_adjust['pos'],
            compound,
        ])
        print(f"  수동 보정 벡터: {emotion_vector}")

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

    with gr.Blocks(title='게임 시나리오 BGM 추천 시스템') as demo:
        gr.Markdown('# 🎮 게임 시나리오 텍스트 기반 BGM 추천 시스템')
        gr.Markdown('시나리오 장면 텍스트를 입력하면 어울리는 BGM을 추천해드려요.')

        with gr.Row():
            with gr.Column(scale=1):
                # 텍스트 입력창
                scene_input = gr.Textbox(
                    label='장면 텍스트',
                    placeholder='예) 적과 마지막 전투를 앞두고 홀로 선 주인공. 두렵지만 앞으로 나아간다.',
                    lines=5,
                )

                gr.Markdown('### 감정 비중 수동 보정')
                gr.Markdown('자동 분석 결과가 마음에 들지 않으면 슬라이더로 조정하세요. (0이면 자동 분석 적용)')

                neg_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label='부정(Negative) 비중')
                pos_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label='긍정(Positive) 비중')
                neu_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label='중립(Neutral) 비중')

                recommend_btn = gr.Button('BGM 추천받기', variant='primary')

            with gr.Column(scale=1):
                # 감정 분석 결과 표시
                emotion_output = gr.Textbox(label='감정 분석 결과', lines=3, interactive=False)

                # 추천 결과 표시
                result_output = gr.Textbox(label='추천 BGM 목록', lines=5, interactive=False)

                # 미리 듣기
                audio_output1 = gr.Audio(label='1순위 BGM 미리 듣기')
                audio_output2 = gr.Audio(label='2순위 BGM 미리 듣기')
                audio_output3 = gr.Audio(label='3순위 BGM 미리 듣기')

        def on_recommend(scene_text, neg_val, pos_val, neu_val):
            if not scene_text.strip():
                return '텍스트를 입력해주세요.', '', None, None, None

            # 수동 보정값 적용 여부 판단
            emotion_adjust = None
            if neg_val > 0 or pos_val > 0 or neu_val > 0:
                total = neg_val + pos_val + neu_val
                if total > 0:
                    emotion_adjust = {
                        'neg': neg_val / total,
                        'pos': pos_val / total,
                        'neu': neu_val / total,
                    }

            results = recommend_bgm(scene_text, emotion_adjust)

            # 감정 분석 결과 텍스트
            bgm_db = load_bgm_db()
            processed = preprocess_text(scene_text)
            sentiment = analyze_sentiment(processed)
            emotion_text = (
                f"부정(neg): {sentiment['neg']:.3f}  "
                f"중립(neu): {sentiment['neu']:.3f}  "
                f"긍정(pos): {sentiment['pos']:.3f}  "
                f"복합(compound): {sentiment['compound']:.3f}"
            )
            if emotion_adjust:
                emotion_text += f"\n수동 보정 적용: neg={emotion_adjust['neg']:.2f}, pos={emotion_adjust['pos']:.2f}, neu={emotion_adjust['neu']:.2f}"

            # 추천 결과 텍스트
            result_lines = []
            for i, r in enumerate(results, 1):
                result_lines.append(f"{i}위. {r['filename']}  (일치도: {r['score']:.1%})")
            result_text = '\n'.join(result_lines)

            # 미리 듣기 파일 경로
            bgm_folder = 'bgm'
            audio_paths = []
            for r in results[:3]:
                path = os.path.join(bgm_folder, r['filename'])
                audio_paths.append(path if os.path.exists(path) else None)

            while len(audio_paths) < 3:
                audio_paths.append(None)

            return emotion_text, result_text, audio_paths[0], audio_paths[1], audio_paths[2]

        recommend_btn.click(
            fn=on_recommend,
            inputs=[scene_input, neg_slider, pos_slider, neu_slider],
            outputs=[emotion_output, result_output, audio_output1, audio_output2, audio_output3],
        )

    demo.launch()


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists("bgm_db.json"):
        print("BGM DB 구축 중...")
        bgm_db = extract_bgm_features("bgm/")
        save_bgm_db(bgm_db)
    else:
        print("기존 BGM DB 불러오는 중...")
        bgm_db = load_bgm_db()
        print(f"BGM DB 로드 완료: {len(bgm_db)}개 트랙")

    build_ui()
