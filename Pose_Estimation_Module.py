import cv2 
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self,mode=False,Complexity=1,smooth_lm=True,enable_seg=False,smooth_seg=True,min_con=0.5,min_tracking=0.5):
        self.mode=mode
        self.complexity=Complexity
        self.smooth_landmarks=smooth_lm
        self.enable_segmentation=enable_seg
        self.smooth_segmentation=smooth_seg
        self.min_detection_confidence=min_con
        self.min_tracking_confidence=min_tracking

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.complexity,self.smooth_landmarks,self.enable_segmentation,self.smooth_segmentation,self.min_detection_confidence,self.min_tracking_confidence)

    def findPose(self,img,draw=True):
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(img_rgb)
        
        if self.results.pose_landmark:
            if draw:
                self.mpDraw.draw.landmarks(img,self.results.pose_landmark,self.mpPose.POSE_CONNECTIONS)
        return img

    
    def findPosition(self,img,draw=True):
        
        self.lmlist=[]

        if self.results.pose_landmark:
            for id, lm in enumerate(self.results.pose_landmark.landmark):
                h ,w ,c=img.shape
                cx, cy =int(lm.x*w),int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                # print(id, cx, cy)
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        return self.lmlist
    
    def findAngle(self,img, p1, p2, p3,draw=True):

        x1 , y1=self.lmlist[p1][1:]
        x2 , y2=self.lmlist[p2][1:]
        x3 , y3=self.lmlist[p3][1:]

        angle=math.degrees(math.atan2(y3-y2,x3-x2)-
                              math.atan2(y1-y2,x1-x2))
        if angle <0:
            angle +=360

        if draw:
            cv2.circle(img,(x1,y1),5,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),5,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),5,(255,0,255),cv2.FILLED)
        
        return angle
        
        



def main():
    cap=cv2.VideoCapture('Videos/1.mp4')
    pTime=0
    detector=poseDetector()

    while True:
        success, img =cap.read()
        img=detector.findPose(img)
        lmlist=detector.findPosition(img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,int(str(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

    


if __name__=="__main__":
    main()