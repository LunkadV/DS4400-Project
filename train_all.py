import subprocess
import sys
import os


PROCESSES = [
    # EDA
    ("results/eda/genre_distribution.png", "Running EDA",
     [sys.executable, "-m", "src.main.eda"]),

    # Preprocessing
    ("data/spectrograms", "Generating spectrograms",
     [sys.executable, "-m", "src.main.models.cnn.generate_spectrograms"]),

    # Training
    ("results/random_forest.json", "Training Random Forest",
     [sys.executable, "-m", "src.main.models.randomForest.train"]),
    ("results/xgboost.json", "Training XGBoost",
     [sys.executable, "-m", "src.main.models.XGBoost.train"]),
    ("results/neural_network.json", "Training Neural Network (MLP)",
     [sys.executable, "-m", "src.main.models.neuralNetwork.train"]),
    ("results/cnn.json", "Training CNN",
     [sys.executable, "-m", "src.main.models.cnn.train"]),

    # Comparison
    ("results/comparison/model_comparison.png", "Generating comparison charts",
     [sys.executable, "-m", "src.main.compare"]),
]


def output_exists(path):
    """Check if a step's output already exists."""
    if os.path.isfile(path):
        return True
    if os.path.isdir(path) and os.listdir(path):
        return True
    return False


if __name__ == "__main__":
    if not os.path.isdir("data/fma_medium"):
        print("ERROR: data/fma_medium/ not found. Download the FMA dataset first.")
        sys.exit(1)

    force = "--force" in sys.argv
    skip_preprocessing = "--skip-preprocessing" in sys.argv

    for output_path, description, cmd in PROCESSES:
        if skip_preprocessing and output_path.startswith("data/"):
            continue

        if output_exists(output_path) and not force:
            print(f"Skipping: {description} (output already exists)")
            continue

        print(f"\n{'=' * 40}")
        print(f"  {description}")
        print(f"{'=' * 40}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: {description} failed (exit code {result.returncode})")
            sys.exit(result.returncode)

    print("\nAll steps complete.")
