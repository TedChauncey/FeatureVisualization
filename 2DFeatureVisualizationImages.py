# FeatureVisualization
### This script allows for deep feature extraction using high performing pretrained
###Survival model. Feature visualization with images using t-SNE, and or PCA
### Author: TC 3/12/18 (happy birthday Watson!)


#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras import backend as K #this just makes stuff work
K.set_image_dim_ordering('th')
from keras.applications.vgg16 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#from other libraries
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chisquare
import scipy.stats as stats

### DATA ###
#input image dimensions
OG_size = 150
img_rows, img_cols = 50, 50 # 50, 50 #normally. make sure this is an even number
Center = OG_size/2
x1, y1, x2, y2 = Center-img_rows/2,Center-img_rows/2,Center+img_cols/2,Center+img_cols/2 # 50, 50, 100, 100

# number of channels
img_channels = 1
# data
Outcomes_file = '/home/chintan/Desktop/AhmedData/Composite.csv' #define the outcomes file, sorted according to PID
#Outcomes_file = '/home/chintan/Desktop/Histology/AllOutcomes.csv'

#path1 = '/home/chintan/Desktop/Histology/ALLImages'

#path1 = '/home/chintan/Desktop/AhmedData/Composite' #change this to only test images    
#path2 = '/home/chintan/Desktop/AhmedData/TestImageCrops'  #DELETE  

path2 = '/home/chintan/Desktop/tsne_python/Crops/Two'

#listing = os.listdir(path1) 
#num_samples=size(listing)
#print num_samples


#for file in listing:
#    im = Image.open(path1 + '/' + file) 
#    img = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
#    #img = im.resize((img_rows,img_cols))
#    gray = img.convert('RGB')
                #need to do some more processing here           
#    gray.save(path2 +'/' +  file, "PNG")

imlist = os.listdir(path2)
imlist.sort() #make sure to SORT the images to match 

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes

Outcomes = pd.read_csv(Outcomes_file) #create seperate outcomes file for TEST data
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'surv2yrTC']) #surv2yrTC, STAGE1_2, HIST3TIER
#PID = pd.Series.as_matrix(Outcomes.loc[:,'patient'])

label = Outcome_of_interest

data,Label = immatrix,label
train_data = [data,Label]

print (train_data[0].shape)
print (train_data[1].shape)

(X, y) = (train_data[0],train_data[1])
X = X.reshape(X.shape[0], img_rows, img_cols,3)
X= X.astype('float32')

X /= 255
#X = preprocess_input(X)


print('X shape:', X.shape)
print(X.shape[0], 'test samples')


### MODEL###

## load pretrained model ##


predDir = '/home/chintan/Desktop/'
modelFile = (os.path.join(predDir,'TheSurvivalModel_VGG16.h5')) #survival based extractor

#modelFile = (os.path.join(predDir,'VGG_ScreeningModel_run5.h5'))

#predDir = '/home/chintan/Desktop/FinalModels'
#modelFile = (os.path.join(predDir,'VGG_SurvivalModel_RightCensored_run2_BESTMODEL_2_6_18.h5')) #survival based extractor
#modelFile = (os.path.join(predDir,'VGG_Abstract_model2.h5'))
#modelFile = (os.path.join(predDir,'VGG_Histopath_run6.h5')) #Histology based extractor 
model = load_model(modelFile)

### Extract features from layer M ###

layer_index = 19 # Set to 19 for 512-D vector, 20 for 4096-D
func1 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[layer_index].output ] )

Feat = np.empty([1,1,512]) #when layer_index =19
#Feat = np.empty([1,1,4096]) # when layer_index =20
for i in xrange(X.shape[0]):
	input_image=X[i,:,:,:].reshape(1,50,50,3)
	input_image_aslist= [input_image]
	func1out = func1(input_image_aslist)
	features = np.asarray(func1out)
	Feat = np.concatenate((Feat, features), axis = 1)

Feat = squeeze(Feat)
Features = Feat[1:Feat.shape[1],:]

### Cluster Analysis ###

#1 Dimension reduction with PCA ##
num_best_features = 2  # number of components after dim reduction

random.seed(10) # set random starting point . 3 and 10 is best
pca = PCA(n_components = num_best_features)
pcam = pca.fit(Features,y)
#Feat = pcam.transform(Features)

F = pcam.transform(Features) ##delete these when using TSNE below

#2 Embedding with TSNE###
from sklearn.manifold import TSNE
#F = TSNE(n_components =2).fit_transform(Feat)

import os
#import matplotlib
#import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


images = X[:,:,:,1].reshape(X.shape[0],2500)

def visualize_scatter_with_images(X_2d_data, images, figsize=(50,50), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.title('Deap learning feature clustering (BLCS dataset Stage 1&2)')
    plt.show()

visualize_scatter_with_images(F, images = [np.reshape(i, (50,50)) for i in images], image_zoom=0.7)
