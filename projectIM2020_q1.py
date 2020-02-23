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
		row = int(height - row)
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
		y1 = int(y0 + 1700*(a)) 
		x2 = int(x0 - 1000*(-b)) 
		y2 = int(y0 - 1000*(a))
	
	return y1



def find_vertices(img, distMax, distMin):
	vertices = []
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(16,16))
	cl = clahe.apply(gray)
	blur = cv2.GaussianBlur(cl,(3,3),0)
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		cnt_len = cv2.arcLength(cnt, True)
		cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
		if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
			cnt = cnt.reshape(-1, 2)
			for i in range(4):
				if(angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4], distMax, distMin)):
					vertices.append(cnt[(i+1) % 4])
	return vertices

	# img = cv2.GaussianBlur(img, (3, 3), 0)
	# for gray in cv2.split(img):
	# 	for thrs in range(0, 255, 25):
	# 		if thrs == 0:
	# 			bin = cv2.Canny(gray, 0, 50, apertureSize=5)
	# 			bin = cv2.dilate(bin, None)
	# 		else:
	# 			_retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
	# 		contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# 		print("contours: {}. type: {}".format(contours, type(contours)))
	# 		for cnt in contours:
	# 			cnt_len = cv2.arcLength(cnt, True)
	# 			cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
	# 			if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
	# 				cnt = cnt.reshape(-1, 2)
	# 				for i in range(4):
	# 					if(angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4], distMax, distMin)):
	# 						vertices.append(cnt[(i+1) % 4])
	# return vertices


def angle_cos(p0, p1, p2, distMax, distMin):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	dis1, dis2 = np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2) ,np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
	arccos = np.arccos(np.dot(d1,d2) / (np.sqrt(d1[0]**2 + d1[1]**2) * np.sqrt(d2[0]**2 + d2[1]**2)))
	if(0 < arccos < 1.70 and (distMin<dis1<distMax) and (distMin<dis2<distMax) ):
		return True
	else:
		return False


def draw_points(vertex, img):
	delta = 10
	height, width, channel = img.shape
	for i in range(-1*delta,delta,1):
		for j in range(-1*delta,delta, 1):
			if vertex[1]+i < height and vertex[0]+j < width:
				img[vertex[1]+i, vertex[0]+j] = [255,0,0]


def remove_redundant_vertices(vertices, steps=15):
	list_vertices = list(map(tuple, np.sort(vertices)))
	print(list_vertices)

	return np.sort(vertices)


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
		vertices = find_vertices(img.copy(), 200, 20)
		output = img.copy()
		for vertex in vertices:
			draw_points(vertex, output)
		plot_results(img, footprint, output)

if __name__ == "__main__":
	main()