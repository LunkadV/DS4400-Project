import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FMA_METADATA_PATH = PROJECT_ROOT / "data" / "fma_metadata"
SPECTROGRAM_PATH = PROJECT_ROOT / "data" / "spectrograms"

def _load_tracks():
    tracks = pd.read_csv(FMA_METADATA_PATH / "tracks.csv", index_col=0, header=[0, 1])
    tracks = tracks[tracks["set", "subset"].isin(["medium", "small"])] # Filter to medium dataset (includes small subset)
    tracks = tracks[tracks["track", "genre_top"].notna()] # Filter out tracks without genre labels
    return tracks

def _encode_labels(tracks):
    encoder = LabelEncoder()
    y = encoder.fit_transform(tracks["track", "genre_top"])
    return y, encoder

def load_tabular():
    tracks = _load_tracks()

    features = pd.read_csv(FMA_METADATA_PATH / "features.csv", index_col=0, header=[0, 1, 2])
    features = features[features.index.isin(tracks.index)]
    tracks = tracks.loc[features.index]

    y, encoder = _encode_labels(tracks)

    X = features.values.astype(np.float32)
    return X, y, encoder

def load_spectrograms():
    tracks = _load_tracks()

    paths = []
    for track_id in tracks.index:
        folder = str(track_id).zfill(6)[:3]
        npy_path = str(SPECTROGRAM_PATH / folder / "arrays" / f"{str(track_id).zfill(6)}.npy")
        paths.append(npy_path)

    y, encoder = _encode_labels(tracks)

    return paths, y, encoder

def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
