from __init__ import *
from datetime import datetime

def generate_image_path() -> str:
    return f"{datetime.now().strftime('%H%M%S')}-{IMAGE_PATH}"