### Students: Ofir Cohen, Saar Weitzman ###
### ID: 312255847, 204175137 ###
### Date: 29/3/2020 ###

############################ IMPORTS ##############################
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
###################################################################


def fix_img_size(img, cut_pixels_left, cut_pixels_right, cut_pixels_top, cut_pixels_bottom):
    '''
    The function gets image pixels to crop and returns the cropped image 
    '''
    new_img = img.copy()
    w, h = img.shape[1], img.shape[0]
    new_img = new_img[cut_pixels_top:h - cut_pixels_bottom, cut_pixels_left:w - cut_pixels_right].copy() # crop the needed piece from the full image

    return new_img


def mse(imgA, imgB):
    '''
	the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    '''
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err  # return the MSE, the lower the error, the more "similar" the two images are


def input_image(img_name, left_margin, right_margin, top_margin, buttom_margin):
    '''
    The function gets an input image, fix it for a better matching and returns it
    '''
    img = cv2.imread("{}".format("images/q3/{}".format(img_name)), 0)
    improved_img = fix_img_size(img, left_margin, right_margin, top_margin, buttom_margin)
    kernel = np.ones((3,3), np.uint8) # kernel for the morphologyEx- to get rid from the noise
    improved_img = cv2.morphologyEx(improved_img, cv2.MORPH_CLOSE, kernel)
    blurred_img = cv2.blur(improved_img, (5,5))
    blurred_img = cv2.blur(blurred_img, (5,5))
    return img, blurred_img


def adjust_db_image_to_input(db_img, img):
    '''
    The function below gets an input image and DB img , make adjustments to the DB image for a better matching and returns it
    '''
    improved_db_img = fix_img_size(db_img, 20, 20, 15, 0) # crop the db image by the crop of the input image
    improved_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0])) # resize the db img we just cropped for the ssim and mse algorithms
    kernel = np.ones((3,3), np.uint8) # kernel for the morphologyEx- to get rid from the noise
    improved_db_img = cv2.morphologyEx(improved_db_img, cv2.MORPH_CLOSE, kernel)
    improved_db_img = cv2.blur(improved_db_img, (7,7))
    improved_db_img = cv2.blur(improved_db_img, (5,5))
    return improved_db_img


def get_input_img(img_num, img_name):
    '''
    The function gets an input image, sends it to the "input_image" function and returns it
    '''
    switcher = {
        0 : input_image(img_name, 50, 0, 6, 0),
        1 : input_image(img_name, 23, 39, 27, 0),
        2 : input_image(img_name, 0, 67, 9, 13),
        3 : input_image(img_name, 40, 40, 21, 0),
        4 : input_image(img_name, 30, 30, 1, 0),
        5 : input_image(img_name, 33, 16, 0, 7)
    }
    return switcher[img_num]


def find_similar_db_image(img, input_num):
    '''
    The function gets the input image and image number in the list of input images, makes the comparison between
    the input image and all the images from the database and returns the save the DB image with the best matching results
    '''
    ssim_max, mse_min, db_closest_img_name = -10, 999999999, ""
    db_closest_img = np.array([])
    DB_size = 1176
    for img_num in range(DB_size):
        current_img_name = get_img_name(img_num)
        db_full_img = cv2.imread("images/q3/DB/{}.png".format(current_img_name), 0)
        db_img = cv2.resize(db_full_img, (img.shape[1], db_full_img.shape[0])) #resize the image to the needed width (height stays the same)

        db_img_to_cmp = db_img[:img.shape[0], :img.shape[1]].copy() # crop the needed piece from the full image

        db_img_to_cmp = adjust_db_image_to_input(db_img_to_cmp, img)

        # ssim = structural similarity index between two images. It returns a value between -1 and 1, when 1 means perfect match and -1 means there no match at all
        s = ssim(img, db_img_to_cmp)

        # mse = mean squared error between the two images. The smaller the value, the better fit there is between the images.
        m = mse(img, db_img_to_cmp)

        if ssim_max < s and mse_min > m - 2500: # Give the mse value less weight than ssim, by letting the mse deviate by 2500  
            ssim_max = s
            mse_min = m
            db_closest_img_name = current_img_name
            db_closest_img = db_full_img

    return ssim_max, mse_min, db_closest_img, db_closest_img_name

######################################## Start of sift solution to q3#############################################################

def showImages(input_img, db_img):
    '''
    The function gets images and displayed them on screen 
    '''
    plt.subplot(121), plt.imshow(input_img, cmap='gray'), plt.title('Input image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(db_img, cmap='gray'), plt.title('DB image')
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_img_name(num):
    '''
    The function gets a number and return back the name of the image need to be pulled next from the DB
    '''
    length, img_name = len(str(num)), ""
    for _ in range(5 - length):
        img_name += "0"

    if num == 0:
        img_name += "0"  # the first picture in database has 6 digits, all the others have 5 digits
    img_name += "{}".format(num)
    return img_name


def get_input_img_for_comparison(img_name):
    '''
    The function gets an input image name, read it, clean it for a better matching and returns it
    '''
    input_img = cv2.imread("images/q3/{}".format(img_name))
    kernel = np.ones((3,3), np.uint8) # kernel for the morphologyEx- to get rid from the noise
    improved_img = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernel)
    blurred_input_img = cv2.blur(improved_img, (5,5))
    return blurred_input_img, input_img


def sift_algorithm(input_img, db_img):
    '''
    The function runs sift algorithm on input_img and db_img, find the number of good points in the comparison and returns that number
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    key_point_1, desc_1 = sift.detectAndCompute(input_img, None)
    key_point_2, desc_2 = sift.detectAndCompute(db_img, None)
    
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    ratio = 0.6 # The ratio helps us to choose the threshold distance between 2 elements
    good_points = []
    for m, n in matches: # m is an element from the first image and n is an element from the second image
        if m.distance < ratio * n.distance: # The lower the distance between those 2 elements, the better is the match
            good_points.append(m)

    # result = cv2.drawMatches(input_img, key_point_1, db_img, key_point_2, good_points, None)
    return good_points


def compare_input_img_with_db_imgs(img_name):
    '''
    The function compares between the input image and all the images in the database. It prints the db_img which had
    the best match to the input_img
    '''
    DB_size = 1176
    best_match_input_img, maximum_good_points = np.array([]), 0

    blurred_input_img, input_img = get_input_img_for_comparison(img_name)

    for img_num in range(DB_size):
        current_img_name = get_img_name(img_num)
        db_img = cv2.imread("images/q3/DB/{}.png".format(current_img_name), 0)

        good_points = sift_algorithm(blurred_input_img, db_img)
        if len(good_points) > maximum_good_points:
            maximum_good_points = len(good_points)
            best_match_input_img = db_img

    print("The number of matches of the most similar image is {}".format(maximum_good_points))
    showImages(input_img, best_match_input_img)


################################################# MAIN ######################################################

if __name__ == "__main__":
    input_imgs_names = ["00001_1.png", "00001_2.png", "00001_3.png", "00471_1.png", "00471_2.png", "00471_3.png"]
    
    for img_name in input_imgs_names:
        compare_input_img_with_db_imgs(img_name)


'''
    input_imgs = []
    for i in range(len(input_imgs_names)):
        input_img, blurred_input_img = get_input_img(i, input_imgs_names[i])
        # plt.imshow(blurred_input_img, cmap='gray'), plt.show() #show the input image after adjusments

        ssim_max, mse_min, db_closest_img, db_closest_img_name = find_similar_db_image(blurred_input_img, i)

        # Save the results of the input images in a list. Each input image results are saved in a dictionary
        input_imgs.append({"input_img" : input_img, "blurred_input_img" : blurred_input_img,
                            "ssim_max" : ssim_max, "mse_min" : mse_min,
                            "db_closest_img" : db_closest_img, "db_closest_img_name" : db_closest_img_name })

        print("The most similar image is {}, with ssim {} and mse {}".format(db_closest_img_name, ssim_max, mse_min))
        showImages(input_img, blurred_input_img, db_closest_img)
'''