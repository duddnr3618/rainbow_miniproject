import cv2
import numpy as np
from fastapi import FastAPI, Request,File, UploadFile,Form,Depends
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from insightface.app import FaceAnalysis
import uvicorn
from models import User,get_db
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from process_image import process_image
from starlette.requests import Request
from binary import binary_image

face_analysis_app = FaceAnalysis(allowed_modules=None, providers=['CPUExecutionProvider'])
face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()

templates = Jinja2Templates(directory="templates")
cap = cv2.VideoCapture(0)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def read_signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request : Request,
    userName: str = Form(...),
    userPhone: int = Form(...),
    profile_pic: UploadFile = File(...),
    db: Session = Depends(get_db),
    
    ):
    # 이미지를 바이너리로 변환
    image_binary = await binary_image(profile_pic)
    
    # 데이터베이스에 저장
    db_user = User(userName=userName, userPhone=userPhone, userImage=image_binary)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/save_image",response_class=HTMLResponse)
async def save_image(file: UploadFile = File(...), db: Session = Depends(get_db),):
    try:
        contents = await file.read()
        # 특정한 파일명으로 업로드된 이미지를 "static/imgs" 디렉토리에 저장
        file_path = f"static/imgs/image.jpg"
        with open(file_path, "wb") as f:
            f.write(contents)

        cap_image = await process_image(contents)
        db_images = await get_all_users_with_images(db)
        db_images = [await process_image(image) for image in db_images]

        for db_image in db_images:
            faces1 = face_analysis_app.get(cap_image)
            faces2 = face_analysis_app.get(db_image)

            feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
            feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
            sims = np.dot(feat1, feat2)

            if sims > 0.55:
                print("Login successful")
                return JSONResponse(content={"success": True})
            else:
                print("Not the same user")
                return JSONResponse(content={"success": False, "error": "Not the same user"})   
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}
  
@app.get("/last", response_class=HTMLResponse)
async def get_last_page(request: Request):
    return templates.TemplateResponse("last.html", {"request": request})

#db에서 image값 가져오는 코드
async def get_all_users_with_images(db: Session):
    users = db.query(User).all()
    user_images = [user.userImage for user in users]
    return user_images

if __name__ == "__main__":
  uvicorn.run('main:app', host='localhost', port=8000, reload=True)
