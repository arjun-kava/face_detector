from facedetector.face import FaceDetector
import cv2


model_path='model/frozen_inference_graph_face.pb'
label_path='model/face_label_map.pbtxt'
align_model='model/shape_predictor_68_face_landmarks.dat'



det=FaceDetector(model_path,label_path,align_model)

image=cv2.imread('5.jpg')

abc=det.get_cordinate(image,align=True)
print(abc)