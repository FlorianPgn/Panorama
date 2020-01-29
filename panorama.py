import cv2
import imutils
import argparse

class Panorama:

	def __init__(self, images):
		self.images = images

	def showSIFT(self, image_idx):
		img = self.images[image_idx]

		sift = cv2.xfeatures2d.SIFT_create()
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		keys, features = sift.detectAndCompute(img, None)
		print(keys)

		
		im = cv2.drawKeypoints(img, keys, img)
		cv2.imshow("Features",im)
		cv2.waitKey(0)


# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--first", required=True,
	help="path to the first image")

args = parser.parse_args()

img = cv2.imread(args.first)

pano = Panorama([img])
pano.showSIFT(0)