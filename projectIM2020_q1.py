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


def find_footprint(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)

	height, width = gray.shape
	row = find_row(edges, height, width)
	if row < height // 2:
		row = int(height - 1.7 * row)
	col = find_col(edges, height, width)
	print("row: {}. col: {}".format(row, col))
	print("height: {}. width: {}".format(height, width))

	return img[:row, col:]




def find_col(img, height, width):
	lines = cv2.HoughLines(img,1,np.pi, 200) 
	
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
	lines = cv2.HoughLines(img,1,np.pi/135, 200) 
	
	for r,theta in lines[0]: 
		a = np.cos(theta) 
		b = np.sin(theta) 
		x0 = a*r 
		y0 = b*r 
		x1 = int(x0 + 1000*(-b)) 
		y1 = int(y0 + 1000*(a)) 
		x2 = int(x0 - 1000*(-b)) 
		y2 = int(y0 - 1000*(a))
	
	print(x1, y1, x2, y2)
	return y1



def find_vertexes(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 3, 5, 0.5)
	img[dst > 0.01 * dst.max()] = [255,0,0]
	return img



def find_rectangles_points(img, distMax, distMin):
	img = cv2.GaussianBlur(img, (3, 3), 0)
	squares = []
	for gray in cv2.split(img):
		for thrs in range(0, 255, 25):
			if thrs == 0:
				bin = cv2.Canny(gray, 0, 50, apertureSize=5)
				bin = cv2.dilate(bin, None)
			else:
				_retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
			contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
				if len(cnt) == 4 and cv2.contourArea(cnt) > 800 and cv2.isContourConvex(cnt):
					cnt = cnt.reshape(-1, 2)
					for i in range(4):
						if(angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4], distMax, distMin)):
							squares.append(cnt[(i+1) % 4])
	return squares


def angle_cos(p0, p1, p2, distMax, distMin):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	dis1, dis2 = np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2) ,np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
	arccos = np.arccos(np.dot(d1,d2) / (np.sqrt(d1[0]**2 + d1[1]**2) * np.sqrt(d2[0]**2 + d2[1]**2)))
	if(0 < arccos < 1.70 and (distMin<dis1<distMax) and (distMin<dis2<distMax) ):
		return True
	else:
		return False


def draw_points (point,img):
	delta = 6
	w,h,d = img.shape
	for i in range(-1*delta,delta,1):
		for j in range(-1*delta,delta, 1):
			if(point[1]+i<w and point[0]+j<h):
				img[point[1]+i,point[0]+j] = [255,0,0]


def plot_results(img, footprint, rectangles):
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
	plt.imshow(rectangles, cmap = 'gray')
	plt.title('Black Rectangles')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def main():
	paths = ['./images/q1/1.jpg', './images/q1/7.JPG']

	for path in paths:
		img = read_img(path)
		footprint1 = find_footprint(img.copy())
		rectangles = find_vertexes(img.copy())
		# rectangles_points = find_rectangles_points(img.copy(), 165, 40)
		# rectangles = img.copy()
		# for point in rectangles_points:
		# 	draw_points(point, rectangles)
		plot_results(img, footprint1, rectangles)

if __name__ == "__main__":
	main()