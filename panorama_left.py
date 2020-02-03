import cv2
import imutils
import argparse
import numpy as np


class Panorama:

	def getImgDesc(self, img):
		sift = cv2.xfeatures2d.SIFT_create()
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		return sift.detectAndCompute(img, None)


	def matchImages(self, img_right, img_left):
		k1, d1 = self.getImgDesc(img_right)
		k2, d2 = self.getImgDesc(img_left)
		#print(k1[0])

		match = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
		matches = match.knnMatch(d1, d2, 2)
		#print(matches[0])
		ratio = 0.6
		accepted_matches = [m for m, n in matches if m.distance < n.distance * ratio]

		return accepted_matches, k1, k2


	def getPano(self, img_right, img_left):

		img_right_flip = cv2.flip(img_right, 1)
		img_left_flip = cv2.flip(img_left, 1)

		img_right = img_left_flip
		img_left = img_right_flip
		
		matches, k1, k2 = self.matchImages(img_right, img_left)

		if len(matches) > 4:
			ptsimg_right = np.array([k1[m.queryIdx].pt for m in matches], dtype="float32")
			ptsimg_left = np.array([k2[m.trainIdx].pt for m in matches], dtype="float32")

		
		#print(img_right.shape[0],img_right.shape[1], img_left.shape[0], img_left.shape[1])
		H, status = cv2.findHomography(ptsimg_right, ptsimg_left, cv2.RANSAC)
		im = cv2.warpPerspective(img_right, H,(img_right.shape[1]+img_left.shape[1], img_right.shape[0]))
		
		im[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
		#cv2.waitKey(0)
		#cv2.imshow("Image 1", img_right)
		#cv2.imshow("Image 2", img_left)
		#cv2.imshow("Pano", im)
		#cv2.waitKey(0)
		return cv2.flip(im, 1)
	
	
		