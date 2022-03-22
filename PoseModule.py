import cv2
import mediapipe as mp

class poseDetector():

	def __init__(self, mode=False, upBody=False, smooth=True,
	             detectionCon=0.5, trackCon=0.5):

		self.mode = mode
		self.upBody = upBody
		self.smooth = smooth
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
		                             self.detectionCon, self.trackCon)

	def findPose(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.pose.process(imgRGB)
		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
				                           self.mpPose.POSE_CONNECTIONS)
		return img


	def findPositions(self, img, draw=True):
		self.lmList = []
		if self.results.pose_landmarks:
			for id, lm in enumerate(self.results.pose_landmarks.landmark):
				h, w, c = img.shape
				# print(id, lm)
				cx, cy = int(lm.x * w), int(lm.y * h)
				self.lmList.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
		return self.lmList


	def findapointposition(self, position):
		print(self.lmList[position][1:])
		return

	def showposition(self, img, p1, draw=True):
		# Get the landmarks
		x1, y1 = self.lmList[p1][1:]
		# Draw
		if draw:
			cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
			cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)