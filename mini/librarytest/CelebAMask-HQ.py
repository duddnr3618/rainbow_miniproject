# from typing import Optional
# from fastapi import FastAPI, File, UploadFile
# import io
# from PIL import Image
# import cv2
# from fastapi.responses import JSONResponse
# import numpy as np

# # Load Face Mesh model
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     static_image_mode=True,
#     max_num_faces=3,
# )

# # Face Mesh drawing utilities
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# # Create FaceAnalysis instance
# face = FaceAnalysis(providers=['CPUExecutionProvider'])
# face.prepare(ctx_id=0, det_size=(640, 640))

# # Load CelebAMask-HQ model
# celebamaskhq_model = CelebAMaskHQ()

# app = FastAPI()

# @app.post("/predict/img2")
# async def predict_api_img(image_file1: UploadFile, image_file2: UploadFile):

#     try:
#         contents1 = await image_file1.read()
#         contents2 = await image_file2.read()

#         buffer1 = io.BytesIO(contents1)
#         buffer2 = io.BytesIO(contents2)

#         pil_img1 = Image.open(buffer1)
#         pil_img2 = Image.open(buffer2)

#         cv_img1 = np.array(pil_img1)
#         cv_img2 = np.array(pil_img2)
#         cv_img1 = cv2.cvtColor(cv_img1, cv2.COLOR_RGB2BGR)
#         cv_img2 = cv2.cvtColor(cv_img2, cv2.COLOR_RGB2BGR)

#         # Face detection with Face Mesh
#         results1 = face_mesh.process(cv_img1)
#         for single_face_landmarks in results1.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image=cv_img1,
#                 landmark_list=single_face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec,
#             )

#         results2 = face_mesh.process(cv_img2)
#         for single_face_landmarks in results2.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image=cv_img2,
#                 landmark_list=single_face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec,
#             )

#         # Save the images with landmarks
#         cv2.imwrite("image1_with_landmarks.jpg", cv_img1)
#         cv2.imwrite("image2_with_landmarks.jpg", cv_img2)

#         # Additional face analysis using CelebAMask-HQ
#         try:
#             # Use the face image for analysis
#             celebamask_result1 = celebamaskhq_model.predict(cv_img1)
#             celebamask_result2 = celebamaskhq_model.predict(cv_img2)

#             # Get the segmentation mask for the face
#             mask1 = celebamask_result1['face_mask']
#             mask2 = celebamask_result2['face_mask']

#             # Apply the pink background filter
#             pink_background = np.zeros_like(cv_img1, dtype=np.uint8)
#             pink_background[:, :] = (255, 182, 193)  # RGB values for pink color

#             filtered_img1 = np.where(mask1[:, :, None] > 0, cv_img1, pink_background)
#             filtered_img2 = np.where(mask2[:, :, None] > 0, cv_img2, pink_background)

#             # Save the images with pink background
#             cv2.imwrite("image1_with_pink_background.jpg", filtered_img1)
#             cv2.imwrite("image2_with_pink_background.jpg", filtered_img2)

#             # ... (rest of the code remains the same)

#     except Exception as e:
#         print("Error processing images:", str(e))
#         return JSONResponse(content={"result": "Error processing images", "similarity": None})