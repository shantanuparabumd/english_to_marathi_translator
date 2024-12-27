from pathlib import Path

# Root of the project (adjust for your specific structure)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT  / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

# Ensure all necessary directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TOKENIZER_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Function to resolve a specific file path
def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path
