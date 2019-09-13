
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math
import os


# In[ ]:





# In[2]:


#image convolve
def co(s_x,pad_image,i):
    img_x = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])
    for x in range (0,height):
        for y in range (0,width):
            su=0
            for k in range (0,3):
                for l in range (0,3):
                    a= pad_image[x-k,y-l]
                    w1 = s_x[k][l];
                    su = su + (w1*a);
            img_x[x,y] = su
        
    print(img_x)
    d = img_x.shape
    print('edge_'+str(i)+'width, height',d)
    gname = 'edge'+str(i)+'.jpg'
    path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2'
    cv2.imwrite(os.path.join(path,gname),np.asarray(img_x))
    cv2.imshow('edge_x',np.asarray(img_x,dtype ='uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.asarray(img_x,dtype='uint8')




# In[9]:


s_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
s_y =[[1,2,1],[0,0,0],[-1,-2,-1]]
#flip
for i in range (0,3):
    temp =s_x[i][0]
    s_x[i][0] =s_x[i][2]
    s_x[i][2] =temp
print('s_x',s_x)
for y in range (0,3):
    temp = s_y[0][y]
    s_y[0][y] =s_y[2][y]
    s_y[2][y] =temp
print('s_y',s_y)

img = cv2.imread('A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\Task\\Task1\\task1.png',0)
im = np.asarray(img)
print(im)
d = img.shape
height = img.shape[0]
width = img.shape[1]
print('Image Dimension    : ',d)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Please Wait... Output is processing')
#g_x = img.copy()
#g_y =img.copy()
# newmatrix
pad_image = np.asarray([[0 for x in range (width+2)] for y in range (height+2)])
pad_image[1:-1,1:-1] = img
cv2.imshow('padded',np.asarray(pad_image,dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
#g_x = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])
g_x = co(s_x,pad_image,1)
cv2.imshow('gx',g_x)
cv2.waitKey(0)
#g_y = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])
g_y = co(s_y,pad_image,2)


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




