from typing import Annotated, Optional
from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
import mediapipe as mp

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

# Load InsightFace ArcFace model
from insightface import model_zoo
arcface_model = model_zoo.get_model('arcface_r100_v1')
arcface_model.prepare(ctx_id=-1)

# Create FaceAnalysis instance
face = FaceAnalysis(providers=['CPUExecutionProvider'])
face.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()

@app.post("/predict/img2")
async def predict_api_img(image_file1: UploadFile, image_file2: UploadFile):

    contents1 = await image_file1.read()
    contents2 = await image_file2.read()

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

    # Save the images with landmarks
    cv2.imwrite("image1_with_landmarks.jpg", cv_img1)
    cv2.imwrite("image2_with_landmarks.jpg", cv_img2)

    # Additional face analysis using DeepFace
    try:
        # Use the face image for analysis
        face_image1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)
        result_list1 = DeepFace.analyze(face_image1, actions=['emotion', 'age', 'gender', 'race'])

        face_image2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2RGB)
        result_list2 = DeepFace.analyze(face_image2, actions=['emotion', 'age', 'gender', 'race'])

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

        # Use InsightFace ArcFace model to extract features
        faces1 = face.get(cv_img1)
        faces2 = face.get(cv_img2)

        feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
        feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
        sims = np.dot(feat1, feat2)
        print(sims)

        if sims > 0.155:
            result = '동일인 입니다'
        else:
            result = '동일인이 아닙니다.'

        return {"result": result, "similarity": float(sims)}

    except Exception as e:
        print("Error in face analysis:", str(e))
        return {"result": "Error", "similarity": None}