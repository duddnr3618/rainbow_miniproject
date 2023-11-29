import cv2
import  base64
import asyncio


connected_clients = set()

async def capture_and_send(websocket):
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        _, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            _, buffer = cv2.imencode('.jpg', face_roi)
            image_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send(image_as_text)
            await asyncio.sleep(0.1)