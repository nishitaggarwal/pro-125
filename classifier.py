import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes  = ['A','B','C','D','E','F','G','H','I','J','K','L','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)


x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 9,
                                                 train_size = 3500,
                                                 test_size = 500 )

x_train_scale = x_train/255.0
x_test_scale = x_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scale,y_train)

y_pred = clf.predict(x_test_scale)
accuracy = accuracy_score(y_test,y_pred)




def get_predicition(image):
    im_PIL = Image.open(image)
    img_bw = im_PIL.convert('L')
    img_bw_resize = img_bw.resize((22,30), Image.ANTIALIAS)
    
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resize,pixel_filter)
    img_bw_resize_scaled = np.clip(img_bw_resize - min_pixel,0,255)
    max_pixel = np.max(img_bw_resize)

    img_bw_resize_inverted_scaled = np.asarray(img_bw_resize_scaled)/max_pixel

    test_sample = np.array(img_bw_resize_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)
    return test_pred[0]