from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ALARM_IMAGE_DIR = OUTPUT_ROOT / "alarm_images"
ALARM_LOG_PATH = OUTPUT_ROOT / "alarm_events.csv"
DEFAULT_DATA_YAML = PROJECT_ROOT / "dataset" / "elevator_ebike.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"


def ensure_runtime_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ALARM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
