import cv2
from imutils import paths
import imutils
import argparse
import numpy as np

import panorama_right as pr
import panorama_left as pl


def multi_v1(imagePaths) :
    for (i, imagePath) in enumerate(imagePaths) :

        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

        if i == 0 :
            img1 = cv2.imread(imagePath)
            continue
        else :
            img2 = cv2.imread(imagePath)    
            img1 = pano_right.getPano(img1, img2)
        
    return img1 

def multi_v2(imagePaths, side) :
    half_imagePaths_size = int(len(imagePaths) / 2)
    is_odd = bool(len(imagePaths) % 2)

    #fin de recursion : half_imagePaths_size = 0
    if half_imagePaths_size == 1 :
        
        img_right = cv2.imread(imagePaths[0])
        img_left = cv2.imread(imagePaths[-1])
        if is_odd :
            img_middle = cv2.imread(imagePaths[1])
            img_right = pano_right.getPano(img_right, img_middle)
            img_left = pano_left.getPano(img_middle, img_left)

        if side == "right" :    
            return pano_right.getPano(img_right, img_left)
        elif side == "left" :
            return pano_left.getPano(img_right, img_left)

    else :
        #séparation imagePaths
        imagePath_right = imagePaths[:half_imagePaths_size]
        imagePath_left = imagePaths[half_imagePaths_size + int(is_odd):]
        if is_odd :
            img_middle = cv2.imread(imagePaths[half_imagePaths_size])
        
        #left
        img_left = multi_v2(imagePath_left, "left")

        #rigth
        img_right = multi_v2(imagePath_right, "right")

        #middle
        if is_odd :
            img_right = pano_right.getPano(img_right, img_middle)
            img_left = pano_left.getPano(img_middle, img_left)
            
            
        if side == "right" :    
            return pano_right.getPano(img_right, img_left)
        elif side == "left" :
            return pano_left.getPano(img_right, img_left)




# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", required=True,
	help="path to the input directory of images")

args = vars(parser.parse_args())

imagePaths = list(paths.list_images(args["directory"]))

#imagePaths.sort(reverse = True)
imagePaths.sort(reverse = True)

pano_right = pr.Panorama()
pano_left = pl.Panorama()


#cv2.imshow("Pano v1", multi_v1(imagePaths)) 

cv2.imshow("Pano v2", multi_v2(imagePaths, "right"))
cv2.waitKey(0)   



#img1 = cv2.imread(args.first)
#img2 = cv2.imread(args.second)


#pano = Panorama([img1, img2])
#pano.showSIFT(0)
#f_matches = pano.getPano(img1, img2)