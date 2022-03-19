import cv2
import mediapipe as mp
import itertools
import numpy as np
import keyboard
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

def find_angle(index1,index2):
  index_list1=list((itertools.chain(*index1)))
  index_list2=list((itertools.chain(*index2)))
  sum1,sum2=0,0
  

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def getSize(image,face_landmarks,indexes):
  imh,imw,_=image.shape
  index_list=list((itertools.chain(*indexes)))
  landmarks=[]
  sumx,sumy,n=0,0,0
  for index in index_list:
    #n+=1
    landmarks.append([int(face_landmarks.landmark[index].x * imw),
                               int(face_landmarks.landmark[index].y * imh)])
    #sumx+=face_landmarks.landmark[index].x
    #sumy+=face_landmarks.landmark[index].y
  landmarks=np.array(landmarks)
  _,_,w,h=cv2.boundingRect(np.array(landmarks))

  return 2*w,2*h,landmarks

def overlay(image,filter_image,face_landmarks,indexes,display=True):
  annotated_img=image.copy()
  try:
    
    face_part_w,face_part_h,landmarks=getSize(image,face_landmarks,indexes)
    resized_filter=cv2.resize(filter_image,(face_part_w,face_part_h))

    _, filter_img_mask = cv2.threshold(resized_filter,
                                           25, 255, cv2.THRESH_BINARY_INV)
    filter_img_mask=cv2.cvtColor(filter_img_mask,cv2.COLOR_BGR2GRAY)
    filter_img_mask=filter_img_mask.astype(np.uint8)
    center=landmarks.mean(axis=0).astype("int")
    
    location = (int(center[0]-face_part_w/2), int(center[1]-face_part_h/2))

    ROI = image[location[1]: int(center[1]+face_part_h/2),
                  location[0]:int(center[0]+face_part_w/2)]

    resultant_image = cv2.bitwise_and(ROI,ROI,mask=filter_img_mask)
    resultant_image = cv2.add(resultant_image,resized_filter)

    annotated_img[location[1]: location[1] + face_part_h,
                      location[0]: location[0] + face_part_w] = resultant_image
  except Exception as e:
    print(e)
    
    
  return annotated_img
    
  


def findFace(cap):
    left_eye = cv2.imread('images/right_eye.jpg')
    right_eye = cv2.imread('images/left_eye.jpg')
  
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            
            img=overlay(image,right_eye,face_landmarks,mp_face_mesh.FACEMESH_RIGHT_EYE)
            final_img=overlay(img,left_eye,face_landmarks,mp_face_mesh.FACEMESH_LEFT_EYE)
          
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(final_img, 1))
        if cv2.waitKey(5) & 0xFF == ord('q'):
          break
      cap.release()
  
  

findFace(cap)
