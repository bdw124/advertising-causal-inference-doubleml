from pathlib import Path
import os
from dotenv import load_dotenv

# load .env file automatically
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR"))