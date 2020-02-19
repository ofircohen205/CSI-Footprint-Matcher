# Name: Ofir Cohen
# ID: 312255847
# Date: 22/3/2020

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img(filePath):
	'''
	Input: image path
	Output: numpy ndarray of the image
	'''
	img = cv2.imread(filePath)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
	imgs = [cv2.imread(file) for file in glob.glob("./images/q3/DB/*.png")]
	paths = ['./images/q3/00001_1.png', './images/q3/00001_2.png', './images/q3/00001_3.png']

	for path in paths:
		img = read_img(path)
		gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp = sift.detect(gray, None)
		img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite('sift_keypoints.jpg',img)



if __name__ == "__main__":
	main()