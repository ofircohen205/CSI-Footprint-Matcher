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



## Part 4:
Find patterns of circles in real crime scene footprints.
We've dealt with the issue with the following steps:
1. Convert RGB image to Grayscale
2. Smooth the image with median filter size 3x3
3. Find the difference between dilation and erosion of an image
4. Use HoughCircles to find patterns of a circles

The outcome for this part is not similar to the outcome in part 2 because the footprints are not "perfect". for example, the image is much more "dirty".