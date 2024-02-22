from __init__ import *
from utils import generate_image_path
from liquid_level import get_water_level
from typing import (
    Union,
    Tuple
)
import os
from flask import (
    Flask,
    request,
    jsonify
)
from werkzeug.datastructures.file_storage import FileStorage
from BottleDetector import BottleDetector
from waitress import serve

app = Flask(__name__)
bottle_detector = BottleDetector(YOLO_PATH)

@app.route("/", methods=["POST"])
def f() -> Union[str, Tuple[str, int]]:
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    image: FileStorage = request.files["image"]

    image_path = generate_image_path()
    image.save(image_path)

    cropped_image = bottle_detector(image_path)
    cropped_image.save("_cr.jpg")

    os.remove(image_path)

    h = cropped_image.size[1]
    cc = round((h - get_water_level(cropped_image)) / h * 20 * 1.26)

    return jsonify({"cc": cc})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=3000)