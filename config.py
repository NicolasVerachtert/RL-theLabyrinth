import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve directory paths from environment variables
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MODELS_DIR = os.getenv("MODELS_DIR", f"{OUTPUT_DIR}/models")
VIDS_DIR = os.getenv("VIDS_DIR", f"{OUTPUT_DIR}/vids")
TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR", f"{OUTPUT_DIR}/tensorboard")

def setup_environment():
    """Ensure required directories exist."""
    for directory in [OUTPUT_DIR, MODELS_DIR, VIDS_DIR, TENSORBOARD_DIR]:
        os.makedirs(directory, exist_ok=True)
