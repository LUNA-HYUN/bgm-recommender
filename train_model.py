"""
DEAM 데이터셋을 활용한 음악 감정 예측 모델 학습

사용법:
    python train_model.py \
        --features  deam/features \
        --annotations1 "deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv" \
        --annotations2 "deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"

DEAM 다운로드:
    https://cvml.unige.ch/databases/DEAM/
"""

import os
import argparse
import numpy as np
import joblib


# ── DEAM features CSV에서 사용할 컬럼 ───────────────────────
# 곡별 CSV의 각 행은 0.5초 단위 값 → 전체 평균으로 집계
FEATURE_COLS = [
    'pcm_RMSenergy_sma_amean',               # rms (에너지)
    'pcm_zcr_sma_amean',                      # zcr (영교차율)
    'pcm_fftMag_spectralCentroid_sma_amean',  # spectral centroid (음색 밝기)
    'pcm_fftMag_spectralFlux_sma_amean',      # spectral flux (음악 변화량)
    'pcm_fftMag_mfcc_sma[1]_amean',          # mfcc 1~13
    'pcm_fftMag_mfcc_sma[2]_amean',
    'pcm_fftMag_mfcc_sma[3]_amean',
    'pcm_fftMag_mfcc_sma[4]_amean',
    'pcm_fftMag_mfcc_sma[5]_amean',
    'pcm_fftMag_mfcc_sma[6]_amean',
    'pcm_fftMag_mfcc_sma[7]_amean',
    'pcm_fftMag_mfcc_sma[8]_amean',
    'pcm_fftMag_mfcc_sma[9]_amean',
    'pcm_fftMag_mfcc_sma[10]_amean',
    'pcm_fftMag_mfcc_sma[11]_amean',
    'pcm_fftMag_mfcc_sma[12]_amean',
    'pcm_fftMag_mfcc_sma[13]_amean',
]


# ── 1단계: DEAM features CSV 로드 ────────────────────────────
def load_deam_features(features_folder: str, song_ids: list) -> dict:
    """
    features/*.csv 파일에서 특성 추출
    각 파일의 행(0.5초 단위)을 평균내어 곡 전체 대표값으로 집계
    """
    import pandas as pd

    result = {}
    total = len(song_ids)

    for i, song_id in enumerate(song_ids):
        filepath = os.path.join(features_folder, f"{song_id}.csv")
        if not os.path.exists(filepath):
            continue

        if i % 100 == 0:
            print(f"  특성 로드: {i}/{total} ({i/total*100:.1f}%)")

        try:
            df = pd.read_csv(filepath, sep=';')

            # 필요한 컬럼만 추출 후 평균
            available = [c for c in FEATURE_COLS if c in df.columns]
            if not available:
                continue

            means = df[available].mean().values
            result[song_id] = means

        except Exception as e:
            print(f"  오류 ({song_id}.csv): {e}")

    return result


# ── 2단계: 어노테이션 로드 ───────────────────────────────────
def load_annotations(*csv_paths: str) -> dict:
    """
    song_level CSV 파일들을 합쳐서 song_id → (valence, arousal) 딕셔너리 반환
    DEAM 스케일 1~9 → valence: -1~1, arousal: 0~1 로 정규화
    """
    import pandas as pd

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # 공백 제거
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  어노테이션 로드: {len(df)}곡")

    result = {}
    for _, row in df.iterrows():
        song_id = int(row['song_id'])
        valence = (float(row['valence_mean']) - 5) / 4   # 1~9 → -1~1
        arousal = (float(row['arousal_mean']) - 1) / 8   # 1~9 →  0~1
        result[song_id] = (valence, arousal)

    return result


# ── 3단계: 데이터셋 구성 ─────────────────────────────────────
def build_dataset(features_dict: dict, annotations_dict: dict):
    """features와 annotations를 song_id 기준으로 매칭"""
    X, y = [], []

    common_ids = set(features_dict.keys()) & set(annotations_dict.keys())
    print(f"  매칭된 곡 수: {len(common_ids)}")

    for song_id in sorted(common_ids):
        X.append(features_dict[song_id])
        valence, arousal = annotations_dict[song_id]
        y.append([valence, arousal])

    return np.array(X), np.array(y)


# ── 4단계: 모델 학습 ─────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray, model_path: str = 'emotion_model.pkl'):
    """
    RandomForest로 음악 특성 → valence/arousal 예측 모델 학습

    데이터 분할 전략:
    - 훈련 80% / 테스트 20% 로 분할
    - 검증 세트는 따로 떼지 않고 5-Fold Cross Validation으로 대체
      (1,800곡 규모에서 검증 세트를 따로 떼면 데이터 낭비가 크기 때문)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.preprocessing import StandardScaler

    print("\n모델 학습 시작...")
    print(f"전체 데이터: {len(X)}곡  특성 수: {X.shape[1]}개")

    # 특성 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5-Fold Cross Validation ──────────────────────────────
    print("\n[5-Fold Cross Validation]")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores, valence_maes, arousal_maes = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_cv.fit(X_tr, y_tr)

        score = model_cv.score(X_val, y_val)
        y_pred = model_cv.predict(X_val)
        valence_mae = np.mean(np.abs(y_pred[:, 0] - y_val[:, 0]))
        arousal_mae = np.mean(np.abs(y_pred[:, 1] - y_val[:, 1]))

        cv_scores.append(score)
        valence_maes.append(valence_mae)
        arousal_maes.append(arousal_mae)
        print(f"  Fold {fold}: R²={score:.3f}  Valence MAE={valence_mae:.3f}  Arousal MAE={arousal_mae:.3f}")

    print(f"\n  평균 R²:          {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    print(f"  평균 Valence MAE: {np.mean(valence_maes):.3f}")
    print(f"  평균 Arousal MAE: {np.mean(arousal_maes):.3f}")

    # ── 최종 모델: 80/20 분할 후 테스트 세트로 최종 평가 ────
    print("\n[최종 모델 학습 — 훈련 80% / 테스트 20%]")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  훈련: {len(X_train)}곡  테스트: {len(X_test)}곡")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    valence_mae = np.mean(np.abs(y_pred[:, 0] - y_test[:, 0]))
    arousal_mae = np.mean(np.abs(y_pred[:, 1] - y_test[:, 1]))

    print(f"  최종 R²:          {score:.3f}")
    print(f"  최종 Valence MAE: {valence_mae:.3f}")
    print(f"  최종 Arousal MAE: {arousal_mae:.3f}")

    # 특성 중요도 출력
    print("\n특성 중요도:")
    for name, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"  {name:45s}: {imp:.3f} {bar}")

    # 모델 + 스케일러 + 특성 목록 저장
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': FEATURE_COLS,
    }, model_path)
    print(f"\n모델 저장 완료: {model_path}")

    return model, scaler


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEAM 기반 감정 예측 모델 학습')
    parser.add_argument('--features', default='deam/features', help='DEAM features 폴더 경로')
    parser.add_argument('--annotations1',
        default='deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')
    parser.add_argument('--annotations2',
        default='deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv')
    parser.add_argument('--model', default='emotion_model.pkl')
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"features 폴더를 찾을 수 없어요: {args.features}")
        exit(1)

    # 어노테이션 로드
    print("어노테이션 로드 중...")
    annotations = load_annotations(args.annotations1, args.annotations2)

    # features 로드
    print("\nDEAM features 로드 중...")
    song_ids = list(annotations.keys())
    features = load_deam_features(args.features, song_ids)

    # 데이터셋 구성
    print("\n데이터셋 구성 중...")
    X, y = build_dataset(features, annotations)

    if len(X) == 0:
        print("데이터가 없어요. 경로를 확인해주세요.")
        exit(1)

    # 학습
    train(X, y, args.model)
    print("\n완료! bgm_recommender.py 실행 시 학습된 모델이 자동으로 적용돼요.")


# ── 1단계: DEAM 음원에서 특성 추출 ───────────────────────────
def extract_single_features(filepath: str) -> dict | None:
    """단일 오디오 파일에서 음악 특성 추출"""
    import librosa

    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True, duration=45)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])

        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))

        return {
            'tempo': tempo,
            'rms': rms,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_contrast': spectral_contrast,
        }
    except Exception as e:
        print(f"  오류 ({os.path.basename(filepath)}): {e}")
        return None


def extract_deam_features(audio_folder: str, annotations_csv: str, cache_path: str = 'deam_features.json'):
    """
    DEAM 전체 데이터셋 특성 추출
    캐시 파일이 있으면 재사용 (시간 절약)
    """
    import pandas as pd

    # 캐시 있으면 로드
    if os.path.exists(cache_path):
        print(f"캐시 파일 발견: {cache_path} → 재사용")
        with open(cache_path) as f:
            cache = json.load(f)
        return np.array(cache['X']), np.array(cache['y'])

    print("DEAM 특성 추출 시작 (시간이 걸릴 수 있어요)...")
    df = pd.read_csv(annotations_csv)

    # DEAM 컬럼명 확인 (버전마다 다를 수 있음)
    print(f"  어노테이션 컬럼: {list(df.columns)}")

    # valence, arousal 컬럼명 자동 탐색
    valence_col = next((c for c in df.columns if 'valence' in c.lower()), None)
    arousal_col = next((c for c in df.columns if 'arousal' in c.lower()), None)
    id_col = next((c for c in df.columns if 'song' in c.lower() or 'id' in c.lower()), None)

    print(f"  사용 컬럼: id={id_col}, valence={valence_col}, arousal={arousal_col}")

    X, y = [], []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        song_id = int(row[id_col])
        filepath = os.path.join(audio_folder, f"{song_id}.mp3")

        if not os.path.exists(filepath):
            continue

        if i % 50 == 0:
            print(f"  진행: {i}/{total} ({i/total*100:.1f}%)")

        features = extract_single_features(filepath)
        if features is None:
            continue

        X.append([
            features['tempo'],
            features['rms'],
            features['zcr'],
            features['spectral_contrast'],
            features['spectral_centroid'],
        ])

        # valence: -1~1, arousal: 0~1 범위로 정규화
        valence = float(row[valence_col])
        arousal = float(row[arousal_col])

        # DEAM은 1~9 스케일인 경우가 있어서 -1~1로 정규화
        if valence > 1:
            valence = (valence - 5) / 4  # 1~9 → -1~1
        if arousal > 1:
            arousal = (arousal - 1) / 8  # 1~9 → 0~1

        y.append([valence, arousal])

    X, y = np.array(X), np.array(y)
    print(f"\n추출 완료: {len(X)}곡")

    # 캐시 저장
    with open(cache_path, 'w') as f:
        json.dump({'X': X.tolist(), 'y': y.tolist()}, f)
    print(f"캐시 저장: {cache_path}")

    return X, y


# ── 2단계: 모델 학습 ─────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray, model_path: str = 'emotion_model.pkl'):
    """
    RandomForest로 음악 특성 → valence/arousal 예측 모델 학습

    데이터 분할 전략:
    - 훈련 80% / 테스트 20% 로 분할
    - 검증 세트는 따로 떼지 않고 5-Fold Cross Validation으로 대체
      (1,800곡 규모에서 검증 세트를 따로 떼면 데이터 낭비가 크기 때문)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor

    print("\n모델 학습 시작...")
    print(f"전체 데이터: {len(X)}곡")

    # 특성 정규화 (전체 데이터 기준으로 먼저 fit)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5-Fold Cross Validation ──────────────────────────────
    print("\n[5-Fold Cross Validation]")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = []
    valence_maes, arousal_maes = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model_cv.fit(X_tr, y_tr)
        score = model_cv.score(X_val, y_val)
        y_pred = model_cv.predict(X_val)

        valence_mae = np.mean(np.abs(y_pred[:, 0] - y_val[:, 0]))
        arousal_mae = np.mean(np.abs(y_pred[:, 1] - y_val[:, 1]))

        cv_scores.append(score)
        valence_maes.append(valence_mae)
        arousal_maes.append(arousal_mae)

        print(f"  Fold {fold}: R²={score:.3f}  Valence MAE={valence_mae:.3f}  Arousal MAE={arousal_mae:.3f}")

    print(f"\n  평균 R²:          {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
    print(f"  평균 Valence MAE: {np.mean(valence_maes):.3f}")
    print(f"  평균 Arousal MAE: {np.mean(arousal_maes):.3f}")

    # ── 최종 모델: 80/20 분할 후 테스트 세트로 최종 평가 ────
    print("\n[최종 모델 학습 — 훈련 80% / 테스트 20%]")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  훈련: {len(X_train)}곡  테스트: {len(X_test)}곡")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 최종 테스트 세트 평가
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    valence_mae = np.mean(np.abs(y_pred[:, 0] - y_test[:, 0]))
    arousal_mae = np.mean(np.abs(y_pred[:, 1] - y_test[:, 1]))

    print(f"  최종 R²:          {score:.3f}")
    print(f"  최종 Valence MAE: {valence_mae:.3f}")
    print(f"  최종 Arousal MAE: {arousal_mae:.3f}")

    # 특성 중요도 출력
    feature_names = ['tempo', 'rms', 'zcr', 'spectral_contrast', 'spectral_centroid']
    importances = model.feature_importances_
    print("\n특성 중요도:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = '█' * int(imp * 30)
        print(f"  {name:20s}: {imp:.3f} {bar}")

    # 모델 + 스케일러 저장
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print(f"\n모델 저장 완료: {model_path}")

    return model, scaler


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEAM 기반 감정 예측 모델 학습')
    parser.add_argument('--audio', default='deam/audio', help='DEAM 오디오 폴더 경로')
    parser.add_argument('--annotations', default='deam/annotations/static_annotations.csv', help='어노테이션 CSV 경로')
    parser.add_argument('--model', default='emotion_model.pkl', help='저장할 모델 경로')
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"오디오 폴더를 찾을 수 없어요: {args.audio}")
        print("DEAM 데이터셋을 먼저 다운로드해주세요: https://cvml.unige.ch/databases/DEAM/")
        exit(1)

    # 특성 추출
    X, y = extract_deam_features(args.audio, args.annotations)

    if len(X) == 0:
        print("추출된 데이터가 없어요. 경로를 확인해주세요.")
        exit(1)

    # 학습
    train(X, y, args.model)
    print("\n완료! 이제 bgm_recommender.py를 실행하면 학습된 모델이 자동으로 적용돼요.")
