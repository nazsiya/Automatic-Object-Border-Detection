import cv2
import numpy as np
from rembg import remove
from PIL import Image

# Read image
image = cv2.imread("Test Image/1.jpg")

# percent by which the image is resized
scale_percent = 25

# calculate the 50 percent of original dimensions
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
img_rez = cv2.resize(image, dsize)

# Select ROI
r = cv2.selectROI("select the area", img_rez)

# Crop image
cropped_image = img_rez[int(r[1]):int(r[1] + r[3]),int(r[0]):int(r[0] + r[2])]

# Display cropped image
cv2.imshow("Cropped image", cropped_image)
cv2.imwrite("output/Crop_img.png",cropped_image)
cv2.imwrite("output/image_rez.png",img_rez)
cv2.waitKey(0)

input_path = 'output/Crop_img.png'
output_path = 'output/output.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)



image_ = cv2.imread('output/output.png')

# Grayscale
gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

blur=cv2.GaussianBlur(gray,(5,5),0)

# Find Canny edges
edged = cv2.Canny(blur,150,450,True)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))
image_copy = cropped_image.copy()
# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(img_rez, contours, -1, (0, 255, 0), 2,offset = [r[0],r[1]])

cv2.imshow('Contours', img_rez)
cv2.imwrite("output/1_out.png",img_rez)
cv2.waitKey(0)
cv2.destroyAllWindows()





