### Students: Ofir Cohen, Saar Weitzman ###
### ID: 312255847, 204175137 ###
### Date: 29/3/2020 ###

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


def find_footprint(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)

	height, width = gray.shape
	row = find_row(edges, height, width)
	if row < height // 2:
		row = int(height - row)
	col = find_col(edges, height, width)
	print("row: {}. col: {}".format(row, col))
	print("height: {}. width: {}".format(height, width))

	return img[:row, col:]


def find_col(img, height, width):
	lines = cv2.HoughLines(img, 1, np.pi, 200) 
	
	for r,theta in lines[0]: 
		a = np.cos(theta) 
		b = np.sin(theta) 
		x0 = a*r 
		y0 = b*r 
		x1 = int(x0 + 1000*(-b)) 
		y1 = int(y0 + 1000*(a)) 
		x2 = int(x0 - 1000*(-b)) 
		y2 = int(y0 - 1000*(a))

	return x1


def find_row(img, height, width):
	lines = cv2.HoughLines(img, 1, np.pi / 135, 200) 
	
	for r,theta in lines[0]: 
		a = np.cos(theta) 
		b = np.sin(theta) 
		x0 = a*r 
		y0 = b*r 
		x1 = int(x0 + 1000*(-b)) 
		y1 = int(y0 + 1700*(a)) 
		x2 = int(x0 - 1000*(-b)) 
		y2 = int(y0 - 1000*(a))
	
	return y1



def find_vertices(img, vertices):
	blur = cv2.GaussianBlur(img, (5,5), 0)
	gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
	threshes = []
	thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
	threshes.append(thresh1)
	thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 0)
	threshes.append(thresh2)

	for thresh in threshes:
		_, contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			approx = cv2.approxPolyDP(cnt, 0.2 * len(cnt), True)
			if len(approx) == 4:
				if cv2.contourArea(cnt) > 9000 and cv2.contourArea(cnt) < 20000:
					vertices.append(cnt)


def draw_points(cnt, img):
	min_width = 99999
	max_width = 0
	min_height = 99999
	max_height = 0
	for points in cnt:
		for point in points:
			if(point[0]< min_width):
				min_width=point[0]
			if (point[0] > max_width):
				max_width = point[0]
			if (point[1] < min_height):
				min_height = point[1]
			if (point[1] > max_height):
				max_height = point[1]

	cv2.circle(img, (min_width, min_height), 5, (255, 0, 0), 10)
	cv2.circle(img, (max_width, min_height), 5, (255, 0, 0), 10)
	cv2.circle(img, (min_width, max_height), 5, (255, 0, 0), 10)
	cv2.circle(img, (max_width, max_height), 5, (255, 0, 0), 10)


def plot_results(img, footprint, output):
	plt.figure(figsize=(20, 20))
	plt.subplot(131)
	plt.imshow(img, cmap='gray')
	plt.title('Original Image')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(132)
	plt.imshow(footprint, cmap = 'gray')
	plt.title('Footprint')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(133)
	plt.imshow(output, cmap = 'gray')
	plt.title('Black Rectangles')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def main():
	paths = ['./images/q1/1.jpg', './images/q1/7.JPG']
	for path in paths:
		img = read_img(path)
		footprint = find_footprint(img.copy())
		vertices = []
		output = img.copy()
		find_vertices(output, vertices)
		for vertex in vertices:
			draw_points(vertex, output)
		plot_results(img, footprint, output)

if __name__ == "__main__":
	main()