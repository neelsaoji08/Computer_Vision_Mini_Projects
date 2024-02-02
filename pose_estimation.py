import cv2 
import mediapipe as mp
import time

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

cap=cv2.VideoCapture('Videos/1.mp4')
pTime=0

while True:
    success, img =cap.read()
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(img_rgb)
    #print(results.pose_landmarks)

    if results.pose_landmark:
        mpDraw.draw.landmarks(img,results.pose_landmark,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmark.landmark):
            h ,w ,c=img.shape
            cx, cy =int(lm.x*w),int(lm.y*h)
            print(id, cx, cy)
            cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)




    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,int(str(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

