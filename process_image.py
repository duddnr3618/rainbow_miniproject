import cv2
import numpy as np
from fastapi import  UploadFile
import io
from PIL import Image


async def process_image(file :bytes):
    buffer = io.BytesIO(file)
    pil_image = Image.open(buffer)
    pil_image_np = np.array(pil_image)
    opencv_image = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)
    return opencv_image