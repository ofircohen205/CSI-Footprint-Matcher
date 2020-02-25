# CSI-Footprint-Matcher

## Overview:
This project was developed during the course 'Image Processing' in Azrieli - Jerusalem College of Engineering.
The project is divided to 4 parts.


## Part 1:
Isolate footprint in image and find the ratio of the b&w rectangle of the ruler
We've dealt with the issue with the following steps:

Isolate footprint:
1. Convert RGB image to Grayscale
2. Execute Canny edge detection on the Grayscaled image
3. Use HoughLines to find the width and height of an sub-image where the footprint exists

Finding the ratio of the B&W rectangles:
1. Convert from RGB to grayscale
2. Smooth
3. Threshold
4. Detect edges
5. Find contours
6. Approximate contours with linear features
7. Find "rectangles" which were structures that: had polygonalized contours possessing 4 points, were of sufficient area, had adjacent edges were ~90 degrees, had distance between "opposite" vertices was of sufficient size, etc.


## Part 2:
Find patterns of circles in footprints.
We've dealt with the issue with the following steps:
1. Convert RGB image to Grayscale
2. Find 3 types of circles - big, medium, small.
    Finding big circles:
    * Find the difference between dilation and erosion of an image
    * Execute local histogram equalization
    * Smooth the image with median filter size 7x7
    * Use HoughCircles to find patterns of a big circles

    Finding medium circles:
    * Execute dilation followed by Erosion
    * Execute local histogram equalization
    * Smooth the image with median filter size 5x5
    * Use HoughCircles to find patterns of a medium circles

    Finding small circles:
    * Execute local histogram equalization
    * Smooth the image with median filter size 5x5
    * Use HoughCircles to find patterns of a small circles


## Part 3:
Simulate a search engine that gets an image of a cropped and noised footprint, and finds the full footprint image of this cropped image, from the Database of footprint images.

We've dealt with the issue with the following steps:
1. Convert RGB cropped image to Grayscale
2. Improve the cropped image by removing the noise and not related parts in it
3. Smooth the cropped image with blur() using filter size 5x5 (on some images we ran it more than one time). On the images that had salt and pepper noise we used morphologyEx() with MORPH_CLOSE and kernel of ones from size 3x3
4. Convert RGB full image from DB to Grayscale
5. Resize the DB image we compare to the width and height of the cropped image (for the ssim check that will come later)
6. Crop the needed area (the upper area of the footprint in that case) from the full DB image for the comparison with the cropped image
7. Smooth the DB image with blur() using filter from the sizes 5x5, 7x7, 9x9 (each cropped input image has a different filter, and on some images we ran it more than one time)
8. Send the improved cropped and DB images to the ssim (=structural similarity index) and mse (=mean squared error) functions, to make the comparison. The ssim is a value between -1 and 1, when 1 means perfect match and -1 means there no match at all. The mse value should be as little as possible. We gave the ssim check more weight than to the mse check in the comparison between the images. 


## Part 4:
Find patterns of circles in real crime scene footprints.
We've dealt with the issue with the following steps:
1. Convert RGB image to Grayscale
2. Smooth the image with median filter size 3x3
3. Find the difference between dilation and erosion of an image
4. Use HoughCircles to find patterns of a circles

The outcome for this part is not similar to the outcome in part 2 because the footprints are not "perfect". for example, the image is much more "dirty".
