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


def find_circles(img, kernel):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	blurred = cv2.medianBlur(gray, 3)
	circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15, param1=100, param2=27, minRadius=10, maxRadius=32)
	output = img.copy()
	draw_circles(output, circles)
	return output


def draw_circles(img, circles):
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for circle in circles[0,:]:
			draw_circle = circle_exists(img, circle[0], circle[1])
			if draw_circle is False:
				center = (circle[0], circle[1])
				radius = circle[2]
				# draw the outer circle
				cv2.circle(img, center, radius, (0,255,0), 3)
				# draw the center of the circle
				cv2.circle(img, center, 1, (0,0,255), 1)


def circle_exists(img, x, y):
	height, width, channel = img.shape
	for row in range(y-15,y+15):
		for col in range(x-15,x+15):
			if row < height and col < width:
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
	paths = ['./images/q4/00004.jpg', './images/q4/00079.png']

	kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

	for path in paths:
		img = read_img(path)
		circle_img = find_circles(img, kernel)
		plot_results(img, circle_img)

if __name__ == "__main__":
	main()