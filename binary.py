import io
from PIL import Image
async def binary_image(file):
    contents = await file.read()
    buffer = io.BytesIO(contents)
    pil_image = Image.open(buffer)
    pil_image = pil_image.convert("RGB")  # Ensure that the image is in RGB format

    with io.BytesIO() as output_buffer:
        pil_image.save(output_buffer, format="JPEG")  # Save the image in JPEG format

        # 처리된 이미지 데이터 반환
        return output_buffer.getvalue()
