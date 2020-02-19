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
	dilate = cv2.dilate(edges, np.ones((10,10), dtype=np.uint8))
	erode = cv2.erode(dilate, np.ones((10,10), dtype=np.uint8))

	index_row = -1
	index_col = -1
	height, width = gray.shape
	lines = cv2.HoughLines(dilate,1,np.pi/180,200)
	for rho,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 500*(-b))
		y1 = int(y0 + 500*(a))
		x2 = int(x0 - 500*(-b))
		y2 = int(y0 - 500*(a))

		print(x1, y1, x2, y2)
		index_col = x1

	row_sums = [0 for _ in range(height)]
	for row in range(height):
		for col in range(200, width):	
			row_sums[row] += erode[row,col]

	max_val = max(row_sums)
	index_row = row_sums.index(max_val)

	# result = erode[index_row:, :] if index_row < (height // 2) else erode[:index_row, :]
	# height, width = result.shape
	# col_sums = [0 for _ in range(width)]
	# for col in range(width):
	# 	for row in range(height):	
	# 		col_sums[col] += result[row,col]

	# max_val = max(col_sums)
	# index_col = col_sums.index(max_val)

	# print("Height: {}. Width: {}".format(gray.shape[0], gray.shape[1]))
	# print("Column: {}. Row: {}".format(index_col, index_row))

	if (index_row < height // 2) and (index_col < width // 2):
		return img[index_row:, index_col:]
	elif (index_row < height // 2) and (index_col > width // 2):
		return img[index_row:, :index_col]
	elif (index_row > height // 2) and (index_col < width // 2):
		return img[:index_row, index_col:]
	else:
		return img[:index_row, :index_col]


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