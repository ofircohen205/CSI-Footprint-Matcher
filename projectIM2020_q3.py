##Saar Weitzman##
##I.D: 204175137##

############################ IMPORT S##############################
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
###################################################################

def improve_img(img, cut_pixels_left, cut_pixels_right, cut_pixels_top, cut_pixels_bottom):
    new_img = img.copy()
    w, h = img.shape[1], img.shape[0]
    new_img = new_img[cut_pixels_top:h - cut_pixels_bottom, cut_pixels_left:w - cut_pixels_right].copy() # crop the needed piece from the full image

    return new_img


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
    print(img_name)
    return img_name


def mse(imgA, imgB):
    '''
	the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    '''
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])

    return err  # return the MSE, the lower the error, the more "similar" the two images are


def showImages(img, blurred_img, edged_img):
    '''
    The function gets images and displayed them on screen 
    '''
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Input image')
    plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(blurred_img, cmap='gray'), plt.title('Input blurred image')
    # plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged_img, cmap='gray'), plt.title('DB image')
    plt.xticks([]), plt.yticks([])
    plt.show()


################## Each function below gets an input image, fix it for a better matching and returns it ################# 
def input_image_471_1():
    img = cv2.imread("{}".format("images/q3/00471_1.png"), 0)
    improved_img = improve_img(img, 11, 10, 18, 0)
    blurred_img = cv2.blur(improved_img, (5,5))
    blurred_img = cv2.blur(blurred_img, (5,5))
    return img, blurred_img


def input_image_471_2():
    img = cv2.imread("{}".format("images/q3/00471_2.png"), 0)
    improved_img = improve_img(img, 0, 0, 1, 10)
    blurred_img = cv2.blur(improved_img, (5,5))
    return img, blurred_img


def input_image_471_3():
    img = cv2.imread("{}".format("images/q3/00471_3.png"), 0)
    improved_img = improve_img(img, 15, 18, 2, 0)

    kernel = np.ones((3,3), np.uint8) # kernel for the morphologyEx- to get rid from the noise
    improved_img = cv2.morphologyEx(improved_img, cv2.MORPH_CLOSE, kernel)
    blurred_img = cv2.blur(improved_img, (5,5))
    return img, blurred_img


def input_image_1_1():
    img = cv2.imread("{}".format("images/q3/00001_1.png"), 0)
    improved_img = improve_img(img, 30, 0, 2, 0)
    blurred_img = cv2.blur(improved_img, (5,5))
    return img, blurred_img


def input_image_1_2():
    img = cv2.imread("{}".format("images/q3/00001_2.png"), 0)
    improved_img = improve_img(img, 15, 15, 18, 0)
    blurred_img = cv2.blur(improved_img, (5,5))
    blurred_img = cv2.blur(blurred_img, (5,5))
    return img, blurred_img


def input_image_1_3():
    img = cv2.imread("{}".format("images/q3/00001_3.png"), 0)
    improved_img = improve_img(img, 0, 35, 9, 0)

    kernel = np.ones((3,3), np.uint8) # kernel for the morphologyEx- to get rid from the noise
    improved_img = cv2.morphologyEx(improved_img, cv2.MORPH_CLOSE, kernel)
    blurred_img = cv2.blur(improved_img, (5,5))
    return img, blurred_img

#######################################################################################################################


######## Each function below gets an input image and DB img , make adjustments to the DB image for a better matching and returns it ################# 
def adjust_db_image_to_input_471_1(db_img, img):
    improved_db_img = improve_img(db_img, 2, 4, 2, 0) # crop the db image by the crop of the input image
    improved_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0])) # resize the db img we just cropped for the ssim and mse algorithms
    blurred_db_img = cv2.blur(improved_db_img, (5,5))
    blurred_db_img = cv2.blur(blurred_db_img, (5,5))
    return blurred_db_img


def adjust_db_image_to_input_471_2(db_img, img):
    improved_db_img = improve_img(db_img, 10, 3, 7, 0) # crop the db image by the crop of the input image
    improved_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0])) # resize the db img we just cropped for the ssim and mse algorithms
    blurred_db_img = cv2.blur(improved_db_img, (7,7))
    blurred_db_img = cv2.blur(blurred_db_img, (7,7))
    return blurred_db_img


def adjust_db_img_to_input_471_3(db_img, img):
    improved_db_img = improve_img(db_img, 0, 20, 20, 0) # crop the db image by the crop of the input image
    cropped_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0]))
    blurred_db_img = cv2.blur(cropped_db_img, (7,7))
    blurred_db_img = cv2.blur(blurred_db_img, (7,7))
    return blurred_db_img


def adjust_db_image_to_input_1_1(db_img, img):
    improved_db_img = improve_img(db_img, 30, 28, 0, 0) # crop the db image by the crop of the input image
    improved_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0])) # resize the db img we just cropped for the ssim and mse algorithms
    improved_db_img = cv2.blur(improved_db_img, (9,9))
    improved_db_img = cv2.blur(improved_db_img, (7,7))
    improved_db_img = cv2.blur(improved_db_img, (7,7))
    return improved_db_img


def adjust_db_image_to_input_1_2(db_img, img):
    improved_db_img = improve_img(db_img, 16, 0, 0, 0) # crop the db image by the crop of the input image
    improved_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0])) # resize the db img we just cropped for the ssim and mse algorithms
    improved_db_img = cv2.blur(improved_db_img, (5,5))
    improved_db_img = cv2.blur(improved_db_img, (5,5))
    return improved_db_img


def adjust_db_image_to_input_1_3(db_img, img):
    improved_db_img = improve_img(db_img, 28, 2, 0, 0) # crop the db image by the crop of the input image
    improved_db_img = cv2.resize(improved_db_img, (img.shape[1], img.shape[0])) # resize the db img we just cropped for the ssim and mse algorithms
    improved_db_img = cv2.blur(improved_db_img, (5,5))
    improved_db_img = cv2.blur(improved_db_img, (5,5))
    return improved_db_img

#######################################################################################################################################################

def get_input_img(img_num):
    switcher = {
        1 : input_image_471_1(),
        2 : input_image_471_2(),
        3 : input_image_471_3(),
        4 : input_image_1_1(),
        5 : input_image_1_2(),
        6: input_image_1_3()
    }
    return switcher[img_num]


def find_similar_db_image(img, input_num):
    ssim_max, mse_min, db_closest_img_name = -10, 999999999, ""
    db_closest_img = np.array([])
    DB_size = 1176
    for img_num in range(DB_size):
        current_img_name = get_img_name(img_num)
        db_full_img = cv2.imread("images/q3/DB/{}.png".format(current_img_name), 0)
        db_img = cv2.resize(db_full_img, (img.shape[1], db_full_img.shape[0])) #resize the image to the needed width (height stays the same)

        db_img_to_cmp = db_img[:img.shape[0], :img.shape[1]].copy() # crop the needed piece from the full image

        if input_num == 1: # image 00471_1
            db_img_to_cmp = adjust_db_image_to_input_471_1(db_img_to_cmp, img)

        elif input_num == 2: # image 00471_2
            db_img_to_cmp = adjust_db_image_to_input_471_2(db_img_to_cmp, img)

        elif input_num == 3: # image 00471_3
            db_img_to_cmp = adjust_db_img_to_input_471_3(db_img_to_cmp, img)
        
        elif input_num == 4: # image 00001_1
            db_img_to_cmp = adjust_db_image_to_input_1_1(db_img_to_cmp, img)

        elif input_num == 5: # image 00001_2
            db_img_to_cmp = adjust_db_image_to_input_1_2(db_img_to_cmp, img)

        else: # input_num = 6, image 00001_3
             db_img_to_cmp = adjust_db_image_to_input_1_3(db_img_to_cmp, img)

        # ssim = structural similarity index between two images. It returns a value between -1 and 1, when 1 means perfect match and -1 means there no match at all
        s = ssim(img, db_img_to_cmp)

        # mse = mean squared error between the two images 
        m = mse(img, db_img_to_cmp)
        print("ssim = {}, mse = {}".format(s, m))

        if ssim_max < s:
            if mse_min > m or mse_min < m + 2000: # Give the mse value less weight than ssim, by letting the mse deviate by 2000
                ssim_max = s
                mse_min = m
                db_closest_img_name = current_img_name
                db_closest_img = db_full_img

    return ssim_max, mse_min, db_closest_img, db_closest_img_name


################################################# MAIN ######################################################

if __name__ == "__main__":
    num_of_inputs = 6
    input_imgs = []

    for i in range(1, num_of_inputs + 1):

        input_img, blurred_input_img = get_input_img(i)

        ssim_max, mse_min, db_closest_img, db_closest_img_name = find_similar_db_image(blurred_input_img, i)

        # Save the results of the input images in a list. Each input image results are saved in a dictionary
        input_imgs.append({"input_img" : input_img, "blurred_input_img" : blurred_input_img,
                            "ssim_max" : ssim_max, "mse_min" : mse_min,
                            "db_closest_img" : db_closest_img, "db_closest_img_name" : db_closest_img_name })

        print("The most similar image is {}, with ssim {} and mse {}".format(db_closest_img_name, ssim_max, mse_min))
        showImages(input_img, blurred_input_img, db_closest_img)

#############################################################################################################