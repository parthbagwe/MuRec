"""
Central configuration for the music recommendation system.
Every path, constant, and hyperparameter lives here — no magic numbers
anywhere else in the codebase.
"""

from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ───────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "music_recommendation_dataset.xlsx"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = ROOT_DIR / "data" / "splits"

TRACKS_CLEAN_PATH = PROCESSED_DIR / "tracks_clean.csv"
AUDIO_MATRIX_PATH = PROCESSED_DIR / "audio_feature_matrix.npy"
LYRIC_MATRIX_PATH = PROCESSED_DIR / "lyric_feature_matrix.npy"
COMBINED_MATRIX_PATH = PROCESSED_DIR / "combined_feature_matrix.npy"

TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"

# ── Model artifact paths ─────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
CONTENT_MODEL_PATH = MODELS_DIR / "content_model.pkl"
COLLAB_MODEL_PATH = MODELS_DIR / "collab_model.pkl"

# ── Column groups ─────────────────────────────────────────────────────────
# These define exactly which columns feed which part of the pipeline.
# Keeping this centralised means if the dataset schema changes, you
# only update it here.

ID_COLUMNS = ["track_id", "title", "artist", "genre", "year"]

AUDIO_FEATURE_COLUMNS = [
    "bpm", "energy", "valence", "brightness",
    "spectral_centroid_hz", "spectral_rolloff_hz", "spectral_flux",
    "rms_energy", "zero_crossing_rate", "tempo_confidence",
] + [f"mfcc_{i}" for i in range(1, 14)] + [
    "chroma_c", "chroma_cs", "chroma_d", "chroma_ds", "chroma_e", "chroma_f",
    "chroma_fs", "chroma_g", "chroma_gs", "chroma_a", "chroma_as", "chroma_b",
]

THEME_POOLS = [
    "loss_longing", "self_identity", "joy_celebration", "nostalgia_memory",
    "anxiety_darkness", "romance_new", "empowerment", "spirituality",
    "social", "bittersweet",
]

LYRIC_FEATURE_COLUMNS = ["primary_theme_pool", "vader_sentiment", "arousal"]

CATEGORICAL_COLUMNS = ["genre", "timbre", "chroma_key", "primary_theme_pool"]

# ── Train / test split ────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_SEED = 42
STRATIFY_COLUMN = "genre"

# ── Hybrid model weights ───────────────────────────────────────────────────
# These three must sum to 1.0. Tune them in Phase 5 based on
# evaluation metrics (Precision@K, NDCG@K).
HYBRID_WEIGHTS = {
    "audio": 0.35,
    "lyric": 0.35,
    "collab": 0.30,
}

# ── Recommendation defaults ────────────────────────────────────────────────
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# ── Collaborative filtering (ALS) hyperparameters ──────────────────────────
ALS_FACTORS = 50
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS = 20

# ── API settings ────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]