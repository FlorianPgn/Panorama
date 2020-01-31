import cv2
import imutils
import argparse
import numpy as np


class Panorama:

	def __init__(self, images):
		self.images = images


	def showSIFT(self, image_idx):
		img = self.images[image_idx]

		keys, features = self.getImgDesc(img)
		
		im = cv2.drawKeypoints(img, keys, img)
		cv2.imshow("Features",im)
		cv2.waitKey(0)


	def getImgDesc(self, img):
		sift = cv2.xfeatures2d.SIFT_create()
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		return sift.detectAndCompute(img, None)


	def matchImages(self, im1, im2):
		k1, d1 = self.getImgDesc(im1)
		k2, d2 = self.getImgDesc(im2)
		print(k1[0])

		match = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
		matches = match.knnMatch(d1, d2, 2)
		print(matches[0])
		ratio = 0.6
		accepted_matches = [m for m, n in matches if m.distance < n.distance * ratio]

		im = self.drawMatches(im1, im2, k1, k2, accepted_matches)

		return accepted_matches, k1, k2


	def drawMatches(self, im1, im2, k1, k2, matches):
		(h1, w1) = im1.shape[:2]
		(h2, w2) = im2.shape[:2]
		im = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
		im[:h1, :w1] = im1
		im[:h2, w1:] = im2
		
		cv2.drawMatches(im1, k1, im2, k2, matches, im, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		
		cv2.imshow("Image", im)
		cv2.waitKey(0)
		
		return im


	def getPano(self, im1, im2):
		matches, k1, k2 = self.matchImages(im1, im2)

		if len(matches) > 4:
			ptsIm1 = np.array([k1[m.queryIdx].pt for m in matches], dtype="float32")
			ptsIm2 = np.array([k2[m.trainIdx].pt for m in matches], dtype="float32")

		
		print(im1.shape[1],im1.shape[1], im2.shape[0], im2.shape[1])
		H, status = cv2.findHomography(ptsIm1, ptsIm2, cv2.RANSAC)
		im = cv2.warpPerspective(im1, H,(im1.shape[1]+im2.shape[1], im1.shape[0]))
		
		im[0:im2.shape[0], 0:im2.shape[1]] = im2
		#cv2.waitKey(0)
		cv2.imshow("Image 1", im1)
		cv2.imshow("Image 2", im2)
		cv2.imshow("Pano", im)
		cv2.waitKey(0)
		


# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--first", required=True,
	help="path to the first image")
parser.add_argument("-s", "--second", required=True,
	help="path to the second image")

args = parser.parse_args()

img1 = cv2.imread(args.first)
img2 = cv2.imread(args.second)


pano = Panorama([img1, img2])
#pano.showSIFT(0)
f_matches = pano.getPano(img1, img2)


