import cv2
import PoseModule as pm

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
detector = pm.poseDetector()
point=input("please select your point: left eye? right eye? left shoulder? right shoulder?")
if point=="left eye":
    position=2
if point=="right eye":
    position=5
if point=="left shoulder":
    position=11
if point=="right shoulder":
    position=12
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPositions(img, False)


    if len(lmList) != 0:
        detector.showposition(img, position)
        detector.findapointposition(position)

    cv2.imshow("Image", img)
    cv2.waitKey(1)