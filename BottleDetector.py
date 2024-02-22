from typing import (
    List,
    Union
)
from ultralytics import YOLO
from ultralytics.engine.results import Results
from PIL.Image import Image
from PIL.Image import open as imgopen
from torch import Tensor

class BottleDetector:
    def __init__(self, model_path: str) -> None:
        self.yolo = YOLO(model_path, task="detect")

    def __call__(self, image: Union[Image, str]) -> Image:
        if isinstance(image, str):
            image = imgopen(image)
        box = self._find_box(image)
        cropped = self._crop_image(image, box)
        return cropped

    def _find_box(self, image: Image) -> List[int]:
        res: Results = self.yolo(image)[0]
        box = res[0].boxes.xyxy.floor().int().tolist()[0]
        return box

    @staticmethod
    def _crop_image(image: Image, box: List[int]) -> Image:
        return image.crop(box)