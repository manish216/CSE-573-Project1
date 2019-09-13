
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

def image_show(image, window_name= 'image'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blur_and_laplacian(image, ksize):
    image_gaussian = cv2.GaussianBlur(image,(ksize,ksize),0)
    #image_show(image_gaussian)
    # [laplacian]
    # Apply Laplace function
    dst = cv2.Laplacian(image_gaussian, cv2.CV_64F, ksize)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst
def template_laplacian(image, ksize):
    dst = cv2.Laplacian(image, cv2.CV_64F, ksize)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst


# In[ ]:


from skimage.feature import match_template

def match_template_opencv_ccoeff_normed(image_gray, temp_gray,image):    
    img_color =image
    img_gray = image_gray
    template = temp_gray

    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    cv.normalize( res, res, 0, 1, cv.NORM_MINMAX, -1 )
    threshold = 0.98
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        #cv.rectangle(image_gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv.rectangle(img_color, pt, (pt[0] + w, pt[1] + h), (232,17,35), 2)
        #rectangle can be made on original image as well
    image_show(img_color)
    
    return image_color


# In[ ]:


import glob
import cv2 as cv
ksize = 7

positive_images = [cv2.imread(file,cv2.IMREAD_COLOR) for file in glob.glob("A:/classes/Intro to Computer vision and Image Processing/projrct/project1/Task3/*.jpg")]
#template = cv2.imread('task3/template15.png',cv2.IMREAD_GRAYSCALE)
template = cv2.imread('A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\Task3\\template15.png',cv2.IMREAD_GRAYSCALE)
b_l_template = template_laplacian(template, ksize)

for image in positive_images:
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    b_l_image = template_laplacian(image, ksize)
    match_template_opencv_ccoeff_normed(b_l_image, b_l_template,image)


# In[ ]:














