import cv2
import numpy as np 
import mediapipe as mp 
import time 
import Hand_Tracking_Module as htm 
import math

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###############################
wCam , hCam = 640,480
##############################
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0


detector=htm.handDetector(detectionCon=0.7)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()

minvol=volRange[0]
maxvol=volRange[1]

volbar=400
while True:

    success ,img =cap.read()
    img= detector.findHands(img)
    lmlist=detector.findPostion(img,draw=False)
    if len(lmlist)!=0:
        #print(lmlist[4],lmlist[8])
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),10,(255,0,225),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,225),cv2.FILLED)
        cv2.circle(img,(cx,cy),10,(255,0,225),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)


        length=math.hypot(x2-x1,y2-y1)
        #print(length)

        if length<=50:
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)

        #Hand Range 50-300
        vol=np.interp(length,[50,250],[minvol,maxvol])
        volbar=np.interp(length,[50,300],[400,150])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)

    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volbar)),(85,400),(0,255,0),cv2.FILLED)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {str(int(fps))}',(20,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)


