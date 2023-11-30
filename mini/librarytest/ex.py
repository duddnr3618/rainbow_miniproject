import cv2
import mediapipe as mp
from deepface import DeepFace

# 얼굴 검출을 위한 객체
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    static_image_mode=True,
    max_num_faces=3,
)

# Face Mesh를 그리기 위한 객체
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 이미지 읽기
image = cv2.imread("face.jpg")

# 얼굴 검출
results = face_mesh.process(image)

# Face Mesh 그리기
for single_face_landmarks in results.multi_face_landmarks:
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=single_face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec,
    )

# 이미지로 저장
cv2.imwrite("face.jpg", image)

# Additional face analysis using deepface
try:
    # Use the face image for analysis
    face_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_list = DeepFace.analyze(face_image, actions=['emotion', 'age', 'gender', 'race'])

    # Iterate over each face result
    for idx, face_result in enumerate(result_list):
        print(f"Face {idx + 1} Analysis:")
        print("Emotion:", face_result['emotion'])
        print("Age:", face_result['age'])
        print("Gender:", face_result['gender'])
        print("Race:", face_result['dominant_race'])
        print("------------------------")

except Exception as e:
    print("Error in face analysis:", str(e))