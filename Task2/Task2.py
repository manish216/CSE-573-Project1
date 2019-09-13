
# coding: utf-8

# In[14]:


##Libraries Used


# In[15]:


import cv2
import numpy as np
import math
import os
import copy


# In[16]:


## Creation of Gaussian Kernel##


# In[17]:


def gaussian_Kernel(x):
    sigma = x
    #print(sigma)
    su = 0.0
    s = 2.0*sigma*sigma
    G_K = [ ]
    for x in range (-3,4): # row
        row =[]
        for y in range(-3,4):# col
           # k =[y,-(x)] #<col,-<row>>
            x1 =  (y) # row
            y1 = -(x) # 
           # print([x,y])
            r = math.sqrt(x1*x1+y1*y1)
            temp = (math.exp(-(r*r)/s))/(math.pi*s)
            su+=temp
            row.append(temp)
        G_K.append(row)            
    #print(G_K)
    #print('sum is ',su)
    return (G_K,su)


# In[18]:


#def oct_list(image):
  #  o = list()
    #img_l =octave_Img(image,G_K1)
    #o.append(np.asarray(octave_Img(image,G_K1))
   # o.append(np.asarray(image))
    #return o


# In[19]:


## Resizing the Image According to the octave


# In[20]:


def resize(image,ctr): 
    img_r = image.copy()
    if(ctr!=0):
        img_r = img_r[::2,::2]
    elif():
        img_r = image.copy()
    return img_r
        
    


# In[21]:


#Covolution Process to create gaussian Images


# In[22]:


def octave_Img(img_p,G_K1): # this is the convolution method which calculate the image and kernel 
    w = G_K1
    img_r = img_p.copy()
    img_2 = img_p.copy()
    d = img_r.shape
    height = img_r.shape[0]
    width  = img_r.shape[1] 
    print('Image Height  :',height)
    print('Image width   :',width)
    x1=0
    id_x =1
    for x in range (3,height-3): # row
        for y in range (3,width-3):# col
            su=0
            for k in range (-3,4):
                for l in range (-3,4):
                    a = img_r.item(x+k,y+l)
                    w1 = w[3+k][3+l];
                    su = su + (w1*a);              
                b = su;
                #img_2.itemset((x,y),b);
                img_2[x,y] =b
    
    
    #ot_list(img_2)
    #return(o.append(np.asarray(image)))
    return (img_2)
    #convolution
    


# In[23]:


# Creating Keypoints


# In[24]:


# keypoints
def keypoint(image,D1,D2,D3,x): ## the function to caluclate the keypoints of the DoG
    for i in range (1,len(np.asarray(D2))-1):
        m_x = D2[i]
        for j in range (1,len(m_x)-1):
            m_y =m_x[j]
            nei =[]
            for k in range(1,-2,-1):
                for l in range(-1,2,1):
                    nei.append(D1[i+1][j+k]) # getting the neighbouring values of D1
                    nei.append(D2[i+1][j+k]) # getting the neighbouring values of D2
                    nei.append(D3[i+1][j+k]) # getting the neighbouring values of D3
            maximum = max(nei)
            minimum = min(nei)
            if((m_y == maximum) or (m_y == minimum)): # checking whether the value os maximum or minimum
                if(x==1 or x==2):
                    print('keypoints for octave 1',[i,j]) 
                    image[i][j] = 255
                elif(x ==3 or x==4):
                    print('keypoints for octave 2',[i,j])
                    image[i*2][j*2] =255
                elif(x==5 or x==6):
                        print('keypoints for octave 3',[i,j])
                    image[i*4][j*4] =255
                elif(x==7 or x ==8):
                    print('keypoints for octave 4',[i,j])
                    image[i*8][j*8] =255        
    return(image)
    
    


# In[25]:


#main Program


# In[26]:


#main Program
img = cv2.imread('A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2.jpg',0)
img_dog = img.copy()
img_k =img.copy()
sigma = (1/(math.sqrt(2)))
res_list =list()
li =list()
img_g = list()
s=1
x=1
id_x =1
for oc in range(0,4):
    sigma1 = sigma
    img_p = resize(img,oc)
    for x in range(0,5):
        (G_K1,Su_N)= gaussian_Kernel(sigma) #calling gaussain kernel
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('gi',np.asarray(G_K1))
        #cv2.waitKey(0)
        img_ga = octave_Img(img_p,G_K1) # calling octave_Img for caluclating the convolution
        #img_g = oct_list(img_p,G_K1,id_x)
        #img_gn = np.abs(img_ga)/np.max(np.abs(img_ga))
        img_g.append(np.asarray(img_ga))
        
        G_name = 'gaussian'+str(id_x)+'.jpg'
        id_x+=1
        path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
        cv2.imwrite(os.path.join(path,G_name),np.asarray(img_ga,dtype='uint8'))
        sigma = sigma * (math.sqrt(2))
    sigma =sigma1
    sigma = sigma * (math.sqrt(2))
    img =img_p.copy()
    res_list.append(np.asarray(img_p))
    #########################################################
    #list
print('=========dog==========')
print(len(img_g))
img_d1 =list()
img_d2 = list()
img_d3 = list()
img_d4 = list()
for i in range(0,4):
    print('start')
    img_dog = img_g[i]-img_g[i+1]
    img_gn = np.abs(img_dog)/np.max(np.abs(img_dog))
    img_d1.append(img_gn)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_dog)
    #cv2.waitKey(0)
    G_name = 'DOG'+str(i)+'.jpg'
    path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
    cv2.imwrite(os.path.join(path,G_name),np.asarray(img_gn,dtype='uint8'))
    print('end') 
 ######################################################################
 # keypoint for octave 1 [for D1,D2,D3]
image0 = keypoint(img_k,img_d1[0],img_d1[1],img_d1[2],1)
G_name = 'k1.jpg'
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
cv2.imwrite(os.path.join(path,G_name),image0)
# keypoint for octave 1 [for D2,D3,D4]
image00 =keypoint(img_k,img_d1[1],img_d1[2],img_d1[3],2)
G_name = 'k2.jpg'
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
cv2.imwrite(os.path.join(path,G_name),image00)

for i in range(5,9):
    print('start')
    img_dog = img_g[i]-img_g[i+1]
    img_gn = np.abs(img_dog)/np.max(np.abs(img_dog))
    img_d2.append(img_gn)
    G_name = 'DOG'+str(i)+'.jpg'
    path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
    cv2.imwrite(os.path.join(path,G_name),np.asarray(img_gn,dtype ='uint8'))
    print('end')
# keypoint for octave 2 [for D1,D2,D3]
image1=keypoint(img_k,img_d2[0],img_d2[1],img_d2[2],3)
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
G_name = 'k3.jpg'
cv2.imwrite(os.path.join(path,G_name),np.asarray(image1,dtype ='uint8'))

# keypoint for octave 2 [for D2,D3,D4]
image2=keypoint(img_k,img_d2[1],img_d2[2],img_d2[3],4)
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
G_name = 'k4.jpg'
cv2.imwrite(os.path.join(path,G_name),np.asarray(image2,dtype ='uint8'))

for i in range(10,14):
    print('start')
    img_dog = img_g[i]-img_g[i+1]
    img_gn = np.abs(img_dog)/np.max(np.abs(img_dog))
    img_d3.append(img_gn)
    G_name = 'DOG'+str(i)+'.jpg'
    path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
    cv2.imwrite(os.path.join(path,G_name),np.asarray(img_gn,dtype ='uint8'))
    print('end')
# keypoint for octave 3 [for D1,D2,D3]
image3 =keypoint(img_k,img_d3[0],img_d3[1],img_d3[2],5)
G_name = 'k5.jpg'
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
cv2.imwrite(os.path.join(path,G_name),np.asarray(image3,dtype ='uint8'))
# keypoint for octave 3 [for D2,D3,D4]
image4=keypoint(img_k,img_d3[1],img_d3[2],img_d3[3],6)
G_name = 'k6.jpg'
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
cv2.imwrite(os.path.join(path,G_name),np.asarray(image4,dtype ='uint8'))
    
for i in range(15,19):
    print('start')
    img_dog = img_g[i]-img_g[i+1]
    img_gn = np.abs(img_dog)/np.max(np.abs(img_dog))
    img_d4.append(img_gn)
    G_name = 'DOG'+str(i)+'.jpg'
    path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
    cv2.imwrite(os.path.join(path,G_name),np.asarray(img_gn,dtype ='uint8'))
    print('end')

# keypoint for octave 4 [for D1,D2,D3]
image5 =keypoint(img_k,img_d4[0],img_d4[1],img_d4[2],7)
G_name = 'k7.jpg'
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
cv2.imwrite(os.path.join(path,G_name),np.asarray(image5,dtype ='uint8'))

# keypoint for octave 4 [for D2,D3,D4]
image6 =keypoint(img_k,img_d4[1],img_d4[2],img_d4[3],8)
G_name = 'k8.jpg'
path = 'A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project1\\task2\\images'
cv2.imwrite(os.path.join(path,G_name),np.asarray(image6,dtype ='uint8'))


# keypoints

    

