import cv2
import mediapipe as mp
import math
import numpy as np
import keyboard


mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2)
dia = 30
rad = int(dia/2)
g=1
x,y=200,0
vel=[0,0]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


with mp_hands.Hands(min_detection_confidence = 0.3, min_tracking_confidence = 0.2) as hand:
    while cap.isOpened():
        ret, frame = cap.read()
        #this is an issue with my cam, delete next line of code if webcam feed is inverted
        frame = cv2.flip(frame, 1)

        image = frame
        results = hand.process(image)
        cv2.rectangle(image,(x,y),(x+50,y+50),(255,0,0),-1)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                
            for id,lm in enumerate(hand_landmark.landmark):
                cx,cy = int(lm.x*480), int(lm.y*640)
                f=[cx,cy]
                #temp=[x,y]
                if id==5:
                    a=[cx,cy]
                if id==8:
                    b=[cx,cy]
                if math.sqrt(((f[0]-x)**2)+((f[1]-y)**2))<=71 and keyboard.is_pressed("w"):
                    x,y=f[0],f[1]
                    vx,vy=(b[0]-a[0]),(b[1]-a[1])
                        
                        
                    if vx!=0 or vy!=0:
                        vel=[vx/math.sqrt(vx**2+vy**2),vy/math.sqrt(vx**2+vy**2)]
                else:
                        if x in range(0,400) and y in range(0,400):
                            x+=int(10*vel[0]/3)
                            y+=int(10*vel[1]/3)
                             
                             
            

        
        cv2.imshow('WebcamFeed', image)

        if y in range(0,400):
            y+=int((10*vel[1]/3)+(g/30))
            g+=1
        else:
            y=399
            vel[1]=vel[1]*(-1)
            g=1
            
 


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()   

 
