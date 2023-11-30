from pyexpat import model
from typing import Optional
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import models
import os
import io
import base64
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles
from PIL import Image
from database import engine, sessionlocal
from sqlalchemy.orm import Session
from database import sessionlocal, engine
import models
import aiofiles


models.Base.metadata.create_all(bind=engine)
model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
model.trainable = False


# Load Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=3,
)

# Face Mesh drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Create FaceAnalysis instance
face = FaceAnalysis(providers=['CPUExecutionProvider'])
face.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()

# Dependency to get the database session
def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()
        

templates = Jinja2Templates(directory="templates")

# Create a directory to store captured images
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Create a directory to serve static files
statics_path = os.path.join("templates", "statics", "images")
if not os.path.exists(statics_path):
    os.makedirs(statics_path)

        
# Example route that uses the database session


# Create a directory to store captured images
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Create a directory to serve static files
statics_path = os.path.join("templates", "statics", "images")
if not os.path.exists(statics_path):
    os.makedirs(statics_path)

async def capture_and_save_image(image: UploadFile):
    # Save the captured image
    image_path = os.path.join(statics_path, "captured_image.jpg")
    
    with open(image_path, "wb") as f:
        f.write(await image.read())  # Use await to read the uploaded file content

    return image_path


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Route to handle both webcam image capture and image upload


@app.post("/upload")
async def upload_image(file: UploadFile = Form(...), webcamName: str = Form(...), phoneNumber: str = Form(...), db: Session = Depends(get_db)):
    try:
        # Read the contents of the uploaded image
        contents = await file.read()

        # Save the uploaded image
        # Use the id (webcamName) as the filename and save it to the specified directory
        image_path = os.path.join("templates", "statics", "images", f"{webcamName}.jpg")
        with open(image_path, "wb") as f:
            f.write(contents)
        isinstance = models.Image_Data(webcamName = webcamName, phoneNumber = phoneNumber, capturedImage = contents)     
            
        db.add(isinstance)
        db.commit()

        return {"filename": f"{webcamName}.jpg", "webcamName": webcamName, "phoneNumber": phoneNumber}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/login")
async def login(webcamName: str = Form(...), file: UploadFile = File(...)):
    try:
        # 업로드된 이미지의 내용을 읽음
        contents = await file.read()
        # 업로드된 이미지 저장
        image_path = os.path.join("captured_images", f"{webcamName}.jpg")
        with open(image_path, "wb") as f:
            f.write(contents)
        image_path1 = os.path.join("templates", "statics", "images", f"{webcamName}.jpg")
        async with aiofiles.open(image_path1, mode='rb') as file:
            contents1 = await file.read()
        image_path2 = os.path.join("C:\hi\dev\playground\mini\captured_images", f"{webcamName}.jpg")
        async with aiofiles.open(image_path1, mode='rb') as file:
            contents2 = await file.read()
        buffer1 = io.BytesIO(contents1)
        buffer2 = io.BytesIO(contents2)
        pil_img1 = Image.open(buffer1)
        pil_img2 = Image.open(buffer2)
        cv_img1 = np.array(pil_img1)
        cv_img2 = np.array(pil_img2)
        cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_RGB2BGR)
        cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_RGB2BGR)
        # InsightFace ArcFace 모델을 사용하여 특징 추출
        faces1 = face.get(cv_img1)
        faces2 = face.get(cv_img2)
        feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
        feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
        sims = np.dot(feat1, feat2)
        print(sims)
        if sims > 0.5:
            return '로그인'
        else:
            return '본인이 아닙니다.'


       
    except Exception as e:
        print(f"오류 발생: {e}")
        return "에러 발생"
    
    
@app.post("/predict/img2")
async def predict_api_img(
    webcamName: str = Form(...)
):
    # try:
        image_path1 = os.path.join("captured_images", f"{webcamName}.jpg")
        image_path2 = os.path.join("templates", "statics", "images", f"{webcamName}.jpg")

        async with aiofiles.open(image_path1, mode='rb') as file:
            contents1 = await file.read()
        
        image_path2 = os.path.join("C:\\hi\\dev\\playground\\mini\\templates\\statics\\images", f"{webcamName}.jpg")
        async with aiofiles.open(image_path2, mode='rb') as file:
            contents2 = await file.read()
        
        buffer1 = io.BytesIO(contents1)
        buffer2 = io.BytesIO(contents2)

        pil_img1 = Image.open(buffer1)
        pil_img2 = Image.open(buffer2)

        cv_img1 = np.array(pil_img1)
        cv_img2 = np.array(pil_img2)
        cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_RGB2BGR)
        cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_RGB2BGR)

        # Face detection with Face Mesh
        results1 = face_mesh.process(cv_img1)
        for single_face_landmarks in results1.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=cv_img1,
                landmark_list=single_face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        results2 = face_mesh.process(cv_img2)
        for single_face_landmarks in results2.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=cv_img2,
                landmark_list=single_face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        # # Save the images with landmarks
        # cv2.imwrite("image1_with_landmarks.jpg", cv_img1)
        # cv2.imwrite("image2_with_landmarks.jpg", cv_img2)


        # Add code to change background color to pink
        pink_background = np.full_like(cv_img1, (193, 182, 255), dtype=np.uint8)
        cv_img1 = np.where(cv_img1 == [0, 0, 0], pink_background, cv_img1)

        pink_background = np.full_like(cv_img2, (193, 182, 255), dtype=np.uint8)
        cv_img2 = np.where(cv_img2 == [0, 0, 0], pink_background, cv_img2)

        # # Save the images with pink background
        # cv2.imwrite("image1_with_pink_background.jpg", cv_img1)
        # cv2.imwrite("image2_with_pink_background.jpg", cv_img2)
    
##kim
        # Additional face analysis using DeepFace
        try:
            # Use the face image for analysis
            face_image1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)
            result_list1 = DeepFace.analyze(face_image1, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

            face_image2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2RGB)
            result_list2 = DeepFace.analyze(face_image2, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

            # Iterate over each face result
            for idx, face_result in enumerate(result_list1):
                print(f"Face {idx + 1} Analysis for Image 1:")
                print("Emotion:", face_result['emotion'])
                print("Age:", face_result['age'])
                print("Gender:", face_result['gender'])
                print("Race:", face_result['dominant_race'])
                print("------------------------")

            for idx, face_result in enumerate(result_list2):
                print(f"Face {idx + 1} Analysis for Image 2:")
                print("Emotion:", face_result['emotion'])
                print("Age:", face_result['age'])
                print("Gender:", face_result['gender'])
                print("Race:", face_result['dominant_race'])
                print("------------------------")

        except Exception as e:
         print("Error in face analysis:", str(e))
         
         
         
         
         
         
         
         
        # If there's an error, return an error message in the response
        # return JSONResponse(content={"result": "Error in face analysis", "similarity": None})
          
##쓸일이 있...

    
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# from fastapi.concurrency import run_in_threadpool
# from wandlab.streamer import Streamer



# # app = FastAPI()
# streamer = Streamer()

# # ... (other routes)

# @app.get('/stream')
# async def stream(src: int = 0):
#     return StreamingResponse(stream_gen(src), media_type="multipart/x-mixed-replace; boundary=frame")

# async def stream_gen(src):
#     try:
#         await run_in_threadpool(streamer.run, src)

#         while True:
#             frame = streamer.bytescode()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     except GeneratorExit:
#         await run_in_threadpool(streamer.stop)