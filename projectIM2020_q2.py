# Name: Ofir Cohen
# ID: 312255847
# Date: 22/3/2020

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


def find_circles(img, kernel1, kernel2):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	big_circles = find_big_circles(gray.copy(), kernel1)
	medium_circles = find_medium_circles(gray.copy(), kernel2)
	little_circles = find_little_circles(gray.copy())
	output = img.copy()
	draw_circles(output, little_circles)
	draw_circles(output, medium_circles)
	draw_circles(output, big_circles)
	return output


def find_big_circles(gray, kernel):
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	#cl = clahe.apply(gradient)
	#blurred = cv2.medianBlur(cl, 7)
	blurred = cv2.blur(gradient, (3,3))
	circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 70, param1=100, param2=55.5, minRadius=17, maxRadius=45)
	return circles


def find_medium_circles(gray, kernel):
	opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# cl = clahe.apply(opening)
	# blurred = cv2.medianBlur(cl, 5)
	blurred = cv2.blur(opening, (3,3))
	circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 48, param1=100, param2=55, minRadius=10, maxRadius=37)
	return circles


def find_little_circles(gray):
	#clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	#cl = clahe.apply(gray)
	#blurred = cv2.medianBlur(cl, 5)
	blurred = cv2.blur(gray, (5,5))
	circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 5, param1=100, param2=33, minRadius=3, maxRadius=20)
	return circles


def draw_circles(img, circles):
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for circle in circles[0,:]:
			center = (circle[0], circle[1])
			radius = circle[2]
			# draw the outer circle
			cv2.circle(img, center, radius, (0,255,0), 3)
			# draw the center of the circle
			cv2.circle(img, center, 1, (0,0,255), 1)


def circle_exists(img, x, y):
	height, width, channel = img.shape
	for row in range(y-30,y+30):
		for col in range(x-30,x+30):
			if img[row,col,0] == 0 and img[row,col,1] == 0 and img[row,col,2] == 255:
				return True
	return False


def plot_results(img, circles):
	plt.figure(figsize=(20, 20))
	plt.subplot(121)
	plt.imshow(img, cmap='gray')
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(circles, cmap = 'gray')
	plt.title('Circles')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def main():
	paths = [
				'./images/q2/00001.png', './images/q2/00002.png', './images/q2/00003.png', './images/q2/00005.png', './images/q2/00006.png',
				'./images/q2/00007.png', './images/q2/00010.png', './images/q2/00471.png', './images/q2/00009.png', './images/q2/00473.png', 
				'./images/q2/00475.png', './images/q2/00476.png', './images/q2/00477.png', './images/q2/00478.png', './images/q2/00480.png'
			]
				

	kernel1 = np.array(	[
							[0,0,1,0,0],
							[0,1,1,1,0],
							[1,1,1,1,1],
							[0,1,1,1,0],
							[0,0,1,0,0]
						], dtype=np.uint8)

	kernel2 = np.array( [
							[0,1,0],
							[1,1,1],
							[0,1,0]
						], dtype=np.uint8)

	for path in paths:
		img = read_img(path)
		circle_img = find_circles(img, kernel1, kernel2)
		plot_results(img, circle_img)


if __name__ == "__main__":
	main()