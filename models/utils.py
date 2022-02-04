
import random
from tqdm import tqdm
import concurrent.futures as cf
from skimage.io import imsave
import time
import cv2
import numpy as np
 
def load_rgb(line,base_path=''):

    size = (512,512)
    rgb = cv2.imread(base_path+line.split(' ')[0])
    if rgb.shape[0]!=size[0] or rgb.shape[1]!=size[1]: 
        rgb = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)

    #print rgb
    rgb = rgb[:,:,::-1]

    return rgb
def load_mask(line,base_path=''):

    size = (512,512)
    mask = cv2.imread(base_path+line.split(' ')[1])
    if mask.shape[0]!=size[0] or mask.shape[1]!=size[1]: 
        mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)

    mask = np.expand_dims(np.mean(mask,axis=-1),axis=-1)

    if 'NIST' in base_path:
        #trans black and white
        mask = 255-mask

    return mask
    
def to_sigmoid(x)



    return np.sigmoid(sig)

def imageLoader(files, batch_size,model):
    size = 512
    L = len(files)
    
    #for .51 gpu groups
    #for i, name in enumerate(files):
    #    files[i]=name.replace('data1','data2')
    #this line is just to make the generator infinite, keras needs that    
    while True:
        random.shuffle(files)
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            '''
            if limit==L:
                break
            X,Y = load_img(files[batch_start:limit])
            '''
            file = files[batch_start:limit]
            #print file
            with cf.ThreadPoolExecutor() as executor:
                rgb = executor.map(load_rgb,file)
                mask = executor.map(load_mask,file)
                rgb = list(rgb)
                mask = list(mask)

                X = np.array(rgb,dtype=np.float32)/255.
                Y = np.array(mask,dtype=np.float32)/255.

            #flip_x,flip_y = flip(X,Y)
            #train_x = np.concatenate(X,flip_x)
            #train_y = np.concatenate(Y,flip_y)
            #yield (train_x,[train_y,train_y]) #a tuple with two numpy arrays with batch_size samples     
            #print 'yield'
            yield (X,[Y,Y])
            batch_start += batch_size   
            batch_end += batch_size


def flip(x,y):
    f = random.randint(0,1)
    flip_x = np.flip(x,f)
    flip_y = np.flip(y,f) 
    return flip_x,flip_y

def rotation(x,y):
    k = random.randint(1,3)
    rot_x = np.rot90(x,k,(1,2))
    rot_y = np.rot90(y,k,(1,2))
    return rot_x,rot_y


def f1_score(y_true,y_pred):
    e = 1e-8
    gp = np.sum(y_true)
    tp = np.sum(y_true*y_pred)
    pp = np.sum(y_pred)
    p = tp/(pp+e)
    r = tp/(gp+e)
    f1 = (2 * p * r) / (p + r + e)
    return f1

