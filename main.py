from typing import Union
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, WebSocket, Form, Request
from fastapi.responses import HTMLResponse,RedirectResponse
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
from database import SessionLocal, engine
import models
from datetime import datetime
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import io 
import os
import base64
from base64 import b64encode

app = FastAPI()

# HTML 파일(템플릿) 위치
templates = Jinja2Templates(directory="templates")

# 데이터베이스 모델 생성
models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 추론기 생성 (task processor 생성)
face = FaceAnalysis(providers=['CPUExecutionProvider'])
face.prepare(ctx_id=0, det_size=(640, 640))

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/img/save", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()

        db_image = models.Timezone(income_time=datetime.now(), outcome_time=datetime.now(), image_binary=contents)
       # db_image = models.UserInfo(income_time=datetime.now(), outcome_time=datetime.now(), image_binary=contents)

        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return HTMLResponse(content=f"이미지 업로드 및 저장 성공 ID: {db_image.id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 업로드 및 저장 실패: {str(e)}")
    

@app.get("/upload")
async def img_inference(request: Request):
    return templates.TemplateResponse ('image_upload.html',{"request" : request})

# @app.post("/uploads")
# async def upload_image(image: dict,db: Session = Depends(get_db)):

#     data_url = image.get("image", "")
#     _, base64_data = data_url.split(",", 1)
#     image_data = base64.b64decode(base64_data)

#     # 로컬 폴더에 이미지 저장
#     with open("uploads/snapshot1.png", "wb") as img_file:
#         img_file.write(image_data)
#         print("로컬 폴더에 저장 ")

#     #return JSONResponse(content={"message": "이미지가 로컬 폴더에 저장되었습니다."})
    

# #@app.post("/img/inference")
# #async def img_inference(image_file1: UploadFile, db: Session = Depends(get_db)):
#     try:
#         # STEP 3 : 추론할 이미지 가져오기 : Get Image (Pre Processing)
#         with open("uploads/snapshot1.png", "rb") as img_file:
#             contents1=img_file.write()
#         # 저장된 이미지들을 가져와 비교
#         stored_images = db.query(models.ImageData).all()
        
#         # 입력 이미지를 NumPy 배열로 변환
#         buffer1 = io.BytesIO(contents1)
#         pil_img1 = Image.open(buffer1)
#         cv_img1 = np.array(pil_img1)
#         cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_RGB2BGR)

#         # 가장 높은 유사도와 해당 이미지 초기화
#         max_similarity = 0.0
#         best_match_image = None

#         for stored_image in stored_images:
#             # 저장된 이미지를 NumPy 배열로 변환
#             buffer2 = io.BytesIO(stored_image.image_binary)
#             pil_img2 = Image.open(buffer2)
#             cv_img2 = np.array(pil_img2)
#             cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_RGB2BGR)

#             # STEP 4 : 추론
#             faces1 = face.get(cv_img1)
#             faces2 = face.get(cv_img2)

#             feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
#             feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
#             sims = np.dot(feat1, feat2)  # dot 행렬 연산식

#             if sims > max_similarity:
#                 max_similarity = sims
#                 best_match_image = stored_image

#         # STEP 5 : Post processing
#         rimg = face.draw_on(cv_img1, faces1)
#         cv2.imwrite("output/iu1.jpg", rimg)

#         rimg = face.draw_on(cv_img2, faces2)
#         cv2.imwrite("output/iu2.jpg", rimg)

#         if max_similarity >= 0.55:
#             # JavaScript 코드를 포함한 HTML 응답 반환
#             return HTMLResponse(content=f"<script>alert('로그인이 성공하였습니다. ID: {best_match_image.id}, 유사도: {max_similarity}');</script>")
#         else:
#             return "출석이 되지 않았습니다. 다시한번 시도해 주세요."

#     except Exception as e:
#         # 오류 처리
#         raise HTTPException(status_code=500, detail=f"이미지 비교 실패: {str(e)}")

@app.post("/uploads")
async def upload_image(username:str , image: dict, db: Session = Depends(get_db)):


    data_url = image.get("image", "")
    _, base64_data = data_url.split(",", 1)
    image_data = base64.b64decode(base64_data)

    # 이미지를 로컬 폴더에 저장
    with open("uploads/snapshot1.png", "wb") as img_file:
        img_file.write(image_data)
        print("로컬 폴더에 저장 ")

    try:
        # STEP 3: 추론할 이미지 가져오기: Get Image (Pre Processing)
        # 'rb' 모드를 사용하여 파일을 읽습니다.
        with open("uploads/snapshot1.png", "rb") as img_file:
            contents1 = img_file.read()

        # 저장된 이미지들을 가져와 비교
        stored_images = db.query(models.ImageData).all()
        stored_images = db.query(models.UserInfo).all()

        # 입력 이미지를 NumPy 배열로 변환
        buffer1 = io.BytesIO(contents1)
        pil_img1 = Image.open(buffer1)
        cv_img1 = np.array(pil_img1)
        cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_RGB2BGR)

        # 가장 높은 유사도와 해당 이미지 초기화
        max_similarity = 0.0
        best_match_image = None

        for stored_image in stored_images:
            # 저장된 이미지를 NumPy 배열로 변환
            buffer2 = io.BytesIO(stored_image.image_binary)
            pil_img2 = Image.open(buffer2)
            cv_img2 = np.array(pil_img2)
            cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_RGB2BGR)

            # STEP 4: 추론
            faces1 = face.get(cv_img1)
            faces2 = face.get(cv_img2)

            feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
            feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
            sims = np.dot(feat1, feat2)  # dot 행렬 연산식

            if sims > max_similarity:
                max_similarity = sims
                best_match_image = stored_image

        # STEP 5: Post processing
        rimg = face.draw_on(cv_img1, faces1)
        cv2.imwrite("output/iu1.jpg", rimg)

        rimg = face.draw_on(cv_img2, faces2)
        cv2.imwrite("output/iu2.jpg", rimg)

        # 성공과 실패에 대한 알림 스크립트 정의
        alert_script_success = """<script>alert('출석이 완료되었습니다. ID: {best_match_image.id}, 유사도: {max_similarity}');</script>"""
        alert_script_failure = """<script>alert('출석이 되지 않았습니다. 다시한번 시도해 주세요.');</script>"""

        # HTMLResponse를 적절한 정보와 함께 반환
        if max_similarity >= 0.55:
            db.query(models.Timezone).filter(models.Timezone.username == username).update({"income_time": datetime.now()})
            db.commit()
            return HTMLResponse(content=alert_script_success, status_code=200)
        else:
            return HTMLResponse(content=alert_script_failure, status_code=400)

    except Exception as e:
        # 오류 처리
        raise HTTPException(status_code=500, detail=f"이미지 비교 실패: {str(e)}")
    

from fastapi import Request
@app.get("/signup")
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup_process(
    username: str = Form(...),
    number: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    request: Request = None,
):
    try:
        # 파일을 읽어서 바이너리로 저장
        contents = await file.read()

        # 이미지 데이터와 다른 회원가입 정보를 데이터베이스에 저장
        db_user = models.UserInfo(username=username, phone_number=number, image_binary=contents)

        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        # 회원가입 성공 시 메인 페이지로 리다이렉트
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"회원가입 실패: {str(e)}")