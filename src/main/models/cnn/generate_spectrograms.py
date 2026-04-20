import librosa
import numpy as np
import os

from src.main.data import PROJECT_ROOT


FMA_MEDIUM_PATH = PROJECT_ROOT / "data" / "fma_medium"
OUTPUT_PATH = PROJECT_ROOT / "data" / "spectrograms"


def generate_spectrogram(source_path, output_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Generates a Mel spectrogram from an audio file and saves it as a numpy array.

    Args:
        source_path (str): Path to the input audio file.
        output_path (str): Path to save the generated spectrogram numpy array.
        sr (int): Sample rate for loading the audio. Default is 22050 Hz.
        n_fft (int): Length of the FFT window. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        n_mels (int): Number of Mel bands to generate. Default is 128.

    Returns:
        bool: True if spectrogram generation and saving was successful, False otherwise.
    """
    try:
        y, sr = librosa.load(source_path, sr=sr)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        np.save(output_path, mel_spectrogram_db)
        return True
    except Exception as e:
        print(f"Error generating spectrogram from {source_path}: {e}")
        return False


def process_folder(source_folder, output_folder, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Processes all audio files in a folder to generate spectrograms as numpy arrays.

    Args:
        source_folder (str): Path to the folder containing input audio files.
        output_folder (str): Path to the folder where generated spectrograms will be saved.
        sr (int): Sample rate for loading the audio. Default is 22050 Hz.
        n_fft (int): Length of the FFT window. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        n_mels (int): Number of Mel bands to generate. Default is 128.
    """
    arrays_folder = os.path.join(output_folder, "arrays")
    os.makedirs(arrays_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(source_folder) if f.endswith(".mp3")]
    failure_count = 0

    for audio_file in audio_files:
        source_path = os.path.join(source_folder, audio_file)
        output_path = os.path.join(arrays_folder, os.path.splitext(audio_file)[0] + ".npy")

        if not generate_spectrogram(source_path, output_path, sr=sr, n_fft=n_fft,
                                    hop_length=hop_length, n_mels=n_mels):
            failure_count += 1

    print(f"Processed {len(audio_files)} files with {failure_count} failures.")


def process_all(sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Processes all subfolders in fma_medium to generate spectrograms.

    Args:
        sr (int): Sample rate for loading the audio. Default is 22050 Hz.
        n_fft (int): Length of the FFT window. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        n_mels (int): Number of Mel bands to generate. Default is 128.
    """
    subfolders = sorted([f.path for f in os.scandir(FMA_MEDIUM_PATH) if f.is_dir()])
    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        print(f"Processing subfolder {folder_name}...")
        output_folder = os.path.join(OUTPUT_PATH, folder_name)
        process_folder(subfolder, output_folder, sr=sr, n_fft=n_fft,
                       hop_length=hop_length, n_mels=n_mels)


if __name__ == "__main__":
    process_all()
