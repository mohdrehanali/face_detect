import cv2,numpy
import time
face_detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detect=cv2.CascadeClassifier("haarcascade_eye.xml")
smile_detect=cv2.CascadeClassifier("haarcascade_smile.xml")
##adding text on webcam
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (5,250)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


video=cv2.VideoCapture(0)
while True:
    check,frame=video.read()
    #time.sleep(2)
    print(frame)
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ##adding image function here
    ##cv2.putText(frame,'Hello World!', 
    ##bottomLeftCornerOfText, 
    ##font, 
    ##fontScale,
    ##fontColor,
    #lineType)
    ##end of fucntion calling
    #print(frame)
    faces=face_detect.detectMultiScale(grey,scaleFactor=1.05,minNeighbors=5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        eye_grey=grey[y:y+h,x:x+w]
        eye_frame=frame[y:y+h,x:x+w]
        eyes=eye_detect.detectMultiScale(eye_grey)
        smile=smile_detect.detectMultiScale(eye_grey)
        print("eye frame is:",eye_frame)
        for a,b,c,d in eyes:
            cv2.rectangle(eye_frame,(a,b),(a+c,b+d),(0,255,0),3)
        for k,l,m,n in smile:
            cv2.rectangle(eye_frame,(k,l),(k+m,l+n),(255,0,0),3)
    cv2.imshow("frame output",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

