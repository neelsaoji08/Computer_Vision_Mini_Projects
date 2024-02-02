import cv2
import mediapipe as mp 
import time 


class FaceMeshDetector():

    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode=static_image_mode
        self.max_num_faces=max_num_faces
        self.refine_landmarks=refine_landmarks
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(self.static_image_mode,self.max_num_faces,self.refine_landmarks,self.min_detection_confidence,self.min_tracking_confidence)
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)


    def findMesh(self, img ,draw=True):

        self.imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceMesh.process(self.imgRGB)
        faces=[]
        if self.results.multi_face_landmarks:
            for facelm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,facelm,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                face=[]
                for id ,lm in enumerate(facelm.landmark):
                    #print(lm)
                    ih,iw,ic=img.shape
                    x , y=int(lm.x*iw),int(lm.y*ih)
                    #print(id, x ,y)
                    face.append([x,y])
                faces.append(face)

        return img ,faces
   


def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    detector=FaceMeshDetector()

    while True:
        success, img=cap.read()
        img, faces=detector.findMesh(img)

        if len(faces)!=0:
            print(len(faces))

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ =="__main__":
    main()