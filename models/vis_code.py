import numpy as np
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201
#from keras.applications.resnext import ResNeXt101
from keras.preprocessing import image
from keras.engine import Layer
import glob
import keras.layers as layers
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Lambda , Input , Reshape , merge, concatenate, Activation, Dense, Dropout, Flatten,ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers.core import RepeatVector, Permute
from keras.layers import MaxPooling2D,AveragePooling2D,UpSampling2D,SeparableConv2D,LeakyReLU,Concatenate,DepthwiseConv2D,Add
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import Callback,ModelCheckpoint,LambdaCallback
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from keras.optimizers import SGD,Adam
import random
import cv2
#from tensorflow.losses import huber_loss 
#from keras_contrib.losses import DSSIMObjective as dssim

from keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score
from skimage.color import rgb2lab,lab2rgb
from skimage.transform import resize
from skimage.measure import compare_ssim,compare_psnr
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures as cf
from skimage.io import imsave
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

gpus = 1


epoch = 15000
dataset = 'defacto'
if dataset == 'defacto': 
    batch_size = gpus*3
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
    base_path = '/data1/zolo/DEFACTO/' 
    #all_files = open(base_path+'all_files.txt','r').read().splitlines()
    splicing  = open(base_path+'splicing.txt','r').read().splitlines()
    copy_move = open(base_path+'copy-move.txt','r').read().splitlines()
    removal = open(base_path+'removal.txt','r').read().splitlines()
    
    #random.shuffle(all_files)
    no_s = int(len(splicing)*0.1)
    no_c = int(len(copy_move)*0.1)
    no_r = int(len(removal)*0.1)

    #for test
    no_ss = int(len(splicing)*0.1)
    no_cc = int(len(copy_move)*0.1)
    no_rr = int(len(removal)*0.1)
    
    train_files = splicing[no_s:]+copy_move[no_c:]+removal[no_r:]
    valid_files = splicing[:no_ss]+copy_move[:no_cc]+removal[:no_rr]
    print len(train_files),len(valid_files)
if dataset == 'tianchi': 
    batch_size = gpus*3
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
    base_path = './'#'/data/zolo/PS_dataset/' 
    all_files = open(base_path+'all_files.txt','r').read().splitlines()
    random.shuffle(all_files)
    no = int(len(all_files)*0.1)
    train_files = all_files[no:]
    valid_files = all_files[:no]
    print len(train_files),len(valid_files)
if dataset == 'ps_dataset': 
    batch_size = gpus*1
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    base_path = './'#'/data/zolo/PS_dataset/' 
    train_files = open(base_path+'train_file.txt','r').read().splitlines()

    valid_files = open(base_path+'test_file.txt','r').read().splitlines()
    print len(train_files),len(valid_files)
if dataset == 'nist':
    batch_size = 66
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,7"
    base_path = '/data1/zolo/forgrey_location/NIST/NC2016_Test/' # NIST2016\_4867.jpg
    nist = open(base_path+'nist.txt','r').read().splitlines()
    train_files = nist[160:]#
    valid_files = nist[:160]
if dataset == 'casia1':
    batch_size = gpus*12
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    base_path = '/data/zolo/forgrey_location/CASIA/'  
    casia1 = open(base_path+'casia1.txt','r').read().splitlines()
    casia2 = open(base_path+'casia2.txt','r').read().splitlines()
    train_files = casia2
    valid_files = casia1
if dataset == 'coverage':
    batch_size = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    base_path = '/data2/forgrey_location/COVERAGE/' 
    #data2/forgrey_location/COVERAGE/coverage.txt
    coverage = open(base_path+'coverage.txt','r').read().splitlines()
    train_files = coverage[25:]
    valid_files = coverage[:25]    
if dataset == 'columbia':
    batch_size = gpus*12
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    base_path = '/data/zolo/forgrey_location/Columbia/' 
    columbia = open(base_path+'columbia.txt','r').read().splitlines()
    no = int(len(columbia)*0.3)
    train_files = columbia[no:]
    valid_files = columbia[:no]           
#val_x = np.load('../forgery_localization/test_data/'+dataset+'_test_x.npy')
#val_y = np.load('../forgery_localization/test_data/'+dataset+'_test_y.npy')
#label_y = np.load('../forgery_localization/test_data/'+dataset+'_label_y.npy')
#val_origin = np.load('../forgery_localization/test_data/'+dataset+'_test_origin.npy')

#print len(valid_files),val_x.shape
def f1_score(y_true,y_pred):
    e = 1e-8
    gp = np.sum(y_true)
    tp = np.sum(y_true*y_pred)
    pp = np.sum(y_pred)
    p = tp/(pp+e)
    r = tp/(gp+e)
    f1 = (2 * p * r) / (p + r + e)
    return f1
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    #y_pred = K.round(y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def f1_2(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
 
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
class RocCallback(Callback):
    def __init__(self,validation_data,save_path):
        #self.x = training_data[0]
        #self.y = training_data[1]
        self.x_val = validation_data[0][0]
        self.y_val = validation_data[0][1]
        #self.test_x,self.test_y = validation_data[1][0],validation_data[1][1]
        
        self.auc1 = 0.5
        self.auc2 = 0.5
        self.f1 = 0
        self.f2 = 0
        self.save_path = save_path
        
        self.label_y = self.y_val#.flatten()
        #self.label_y = np.round(np.mean(self.y_val,axis=-1))
        #self.label_y = (1-self.label_y).flatten()
        #self.test_label = np.round(np.mean(self.test_y,axis=-1))
        #self.test_label = (1-self.test_label).flatten()
       
        print 'success in load the data'

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #out = self.model.predict(self.test_x,verbose=1)
        standard_out = self.model.predict(self.x_val,verbose=1,batch_size=batch_size)
        
        sout1,sout2 = standard_out[0],standard_out[1] 

        #slabel_out1 = np.round(np.mean(sout1,axis=-1))
        #slabel_out1 = 1 - slabel_out1

        #slabel_out2 = np.round(np.mean(sout2,axis=-1))
        #slabel_out2 = 1-slabel_out2
        

        
        auc1 = 0#roc_auc_score(self.label_y,sout1.flatten())
        auc2 = 0#roc_auc_score(self.label_y,sout2.flatten())
        f1 = f1_score(self.label_y,sout1)
        f2 = f1_score(self.label_y,sout2)
        
        #print 'standard auc:',sauc1,sauc2        
        
        
        #out1,out2 = np.round(out[0]),np.round(out[1])
        
        #label_out1 = np.round(np.mean(out1,axis=-1))
        #label_out1 = 1 - label_out1

        #label_out2 = np.round(np.mean(out2,axis=-1))
        #label_out2 = 1-label_out2

 

        #auc1 = roc_auc_score(self.test_label,label_out1.flatten())
        #auc2 = roc_auc_score(self.test_label,label_out2.flatten())
        
        print 'results-auc:',auc1,auc2,'f1-score:',f1,f2
        flag = True
        if auc1>self.auc1:
            flag = True
            self.auc1 = auc1
        if auc2>self.auc2:
            flag = True
            self.auc2 = auc2
        if f1>self.f1:
            self.f1 = f1
            flag = True
        if f2>self.f2:
            self.f2 = f2
            flag = True
        for i in range(5):
            try:
                imsave(self.save_path+'images/'+str(i)+'forgery.png',self.x_val[i])
                imsave(self.save_path+'images/'+str(i)+'mask.png',(self.y_val[i,:,:,0]*255).astype(np.uint))
                imsave(self.save_path+'images/'+str(i)+'out1.png',(sout1[i,:,:,0]*255).astype(np.uint))
                imsave(self.save_path+'images/'+str(i)+'out2.png',(sout2[i,:,:,0]*255).astype(np.uint))
            except:
                print 'something error'
                pass
        if flag:
            save_name = self.save_path+str(epoch+1)+'_auc1:'+str(auc1)+'_auc2:'+str(auc2)+'_f1:'+str(f1)+'f2:'+str(f2)+'.h5'
            print save_name
        
            if gpus>1:
                self.model.get_layer('model_1').save(save_name)
            else:
                self.model.save(save_name)
        #print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        
        return
        

def srm_init(shape,dtype=None):

    hpf = np.zeros(shape,dtype=np.float32)
    
    hpf[:,:,0,0]= np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]])/4.0
    hpf[:,:,0,1]= np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]])/12.
    hpf[:,:,0,2]= np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]])/2.0    
    
    return hpf


  
class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        origin = input
        input = DepthwiseConv2D(depth_multiplier=3,kernel_size=(5,5),padding='same',depthwise_initializer=srm_init)(input)
        input.trainable=False
        input = Conv2D(256, 1, use_bias=False, kernel_initializer='he_normal')(input)
        
        input_shape = origin.get_shape().as_list()
        _, h, w, filters = input_shape
        h = tf.shape(origin)[1]
        w = tf.shape(origin)[2]
        
        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input) 

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('sigmoid')(bcT) 
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + origin
        return out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        origin = input
        input = DepthwiseConv2D(depth_multiplier=3,kernel_size=(5,5),padding='same',depthwise_initializer=srm_init)(input)
        input.trainable=False
        input = Conv2D(256, 1, use_bias=False, kernel_initializer='he_normal')(input)
        
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape
        h = tf.shape(origin)[1]
        w = tf.shape(origin)[2]

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('sigmoid')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma*aaTa + origin
        return out    
    
def vgg_block(x,filters,pooling=False,is_seven=False,last=False,name='out1'):
    #2*(3x3)= 5x5  3*(3*3) = 7x7
    
    
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    if is_seven:
        x = layers.Conv2D(filters, (3, 3),
                        padding='same')(x)
        if last:
            x = Activation('sigmoid',name=name)(x)
        else:
            x = Activation('relu')(x)
    if pooling:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x


def det_deconv(x,filters):
    x = UpSampling2D()(x)
    x = Conv2D(filters,3,activation='relu',padding='same')(x)
    return x
   
def auc(y_true, y_pred):
    #y_p = tf.ceil(y_pred, name=None)
    #y_true = layers.Average()([y_true])
    #y_pred = layers.Average([y_pred])
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    y_true = K.flatten(K.mean(y_true,axis=-1)) 
    y_pred = K.flatten(K.mean(y_pred,axis=-1))

    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    #auc = roc_auc_score(y_true,y_pred)
    return auc



def dis():
    def conv(x,filters,kernel=5,strides=2,dilation=1):
        x = Conv2D(filters=filters,kernel=kernel,strides=strides,dilation_rate=dilation,padding='same')(x)
        x = LeakyReLU()(x)
        return x
    dis_in = Input(shape=(256,256,3))
    x = conv(dis_in,64)
    x1 = conv(x,128)
    x2 = conv(x1,256)
    x3 = conv(x2,512)
    x4 = Flatten()(x3)  
    x4 = Dense(1)(x4)
    model = Model(dis_in,x4)

def GAN(g,d):
   
    block_input = Input(shape=(256,256,3),name='gan_input_2')
    tf.get_variable_scope().reuse_variables()
    [out1,out2,coarse,fine] = g([block_input])
    d.trainable = False    
    #concatenate

    out = d(fine)
    
    
    model = Model(inputs=[block_input],outputs=[out1,out2,coarse,fine,out])
    print(model.summary())  
    model.compile(loss=['binary_crossentropy','binary_crossentropy',mix_loss,mix_loss,mix_loss],
                  #loss_weights=[1,1000],
                    optimizer='adam',
                    metrics=[auc,auc,psnr,psnr,'acc'])
    print model.metrics_names
    
    return model
    
def load_img(filelines):
    X = []
    Y = []
    O = []
    size = (512,512)
    for line in filelines:
        #origin = cv2.imread(base_path+line.split(' ')[0])
        #origin = cv2.resize(origin,size,interpolation=cv2.INTER_AREA)
        #origin = origin[:,:,::-1]
        try:
            rgb = cv2.imread(base_path+line.split(' ')[0])
            rgb = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
            rgb = rgb[:,:,::-1]

            mask = cv2.imread(base_path+line.split(' ')[1])#,cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(np.mean(mask,axis=-1),axis=-1)
        except:
            continue
        #for tip-dataset the forgery is 0 so change it
        #np.round(np.mean(self.y_val,axis=-1))
        if 'NIST' in base_path:
            #trans black and white
            mask = 255-mask
        
        
        X.append(rgb/255.)
        Y.append(mask/255.)
        #O.append((origin/255.)*2.0-1)
    
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)#,np.array(O,dtype=np.float32)
from keras.losses import binary_crossentropy
import gc
def fgsm( image, y_true,eps=.01):
    #image = tf.image.decode_image(image)
    if len(image.shape)==3:
        image = np.expand_dims(image,axis=0)
        y_true = np.expand_dims(y_true,axis=0)
    adv_img = np.copy(image)
    

    grads = _grads([adv_img]) 
    grads = np.swapaxes(np.array(grads), 0, 1)
    #- for the adverasarial attack
    grads = -grads.reshape(image.shape)
    
    gradient_norm = np.sign(grads)
    
    adv_imgs = adv_img + eps * gradient_norm
    
    adv_imgs = np.clip(adv_imgs, 0, 1) 
    #sess.graph.finalize()

    #q.put(adv)
    #del y_pred,grads_logits,_grads,grads,gradient_norm
    #gc.collect()
    return adv_imgs
def flip(x,y):
    flip_x = np.flip(x,1)
    flip_y = np.flip(y,1)
    
    return flip_x,flip_y
def imageLoader(files, batch_size,model):
    size = 256
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
            
            yield (X,[Y,Y])
            batch_start += batch_size   
            batch_end += batch_size
            


class TestCallback():
    def __init__(self,validation_data,save_path,model):
        #self.x = training_data[0]
        #self.y = training_data[1]
        self.x_val = validation_data[0][0]
        self.y_val = validation_data[0][1]
        #self.test_x,self.test_y = validation_data[1][0],validation_data[1][1]
        
        self.auc1 = 0
        self.auc2 = 0
        self.f1 = 0
        self.f2 = 0
        self.save_path = save_path
        
        self.label_y = self.y_val.flatten()
        self.model = model
        #self.label_y = np.round(np.mean(self.y_val,axis=-1))
        #self.label_y = (1-self.label_y).flatten()
        #self.test_label = np.round(np.mean(self.test_y,axis=-1))
        #self.test_label = (1-self.test_label).flatten()
       
        print 'success in load the data'


    def test(self, epoch):
        #out = self.model.predict(self.test_x,verbose=1)
        
        standard_out = self.model.predict(self.x_val,verbose=1,batch_size=batch_size)
        
        sout1,sout2 = standard_out[0],standard_out[1] 

        #slabel_out1 = np.round(np.mean(sout1,axis=-1))
        #slabel_out1 = 1 - slabel_out1

        #slabel_out2 = np.round(np.mean(sout2,axis=-1))
        #slabel_out2 = 1-slabel_out2
        

        
        auc1 = 0#roc_auc_score(self.label_y,sout1.flatten())
        auc2 = 0#roc_auc_score(self.label_y,sout2.flatten())
        f1 = f1_score(self.label_y,sout1.flatten())
        f2 = f1_score(self.label_y,sout2.flatten())
        
        #print 'standard auc:',sauc1,sauc2        
        
        
        #out1,out2 = np.round(out[0]),np.round(out[1])
        
        #label_out1 = np.round(np.mean(out1,axis=-1))
        #label_out1 = 1 - label_out1

        #label_out2 = np.round(np.mean(out2,axis=-1))
        #label_out2 = 1-label_out2

 

        #auc1 = roc_auc_score(self.test_label,label_out1.flatten())
        #auc2 = roc_auc_score(self.test_label,label_out2.flatten())
        
        print 'results-auc:',auc1,auc2,'f1-score:',f1,f2
        flag = True
        if auc1>self.auc1:
            flag = True
            self.auc1 = auc1
        if auc2>self.auc2:
            flag = True
            self.auc2 = auc2
        if f1>self.f1:
            self.f1 = f1
            flag = True
        if f2>self.f2:
            self.f2 = f2
            flag = True
        for i in range(5):
            imsave(self.save_path+'images/'+str(i)+'forgery.png',self.x_val[i])
            imsave(self.save_path+'images/'+str(i)+'mask.png',(self.y_val[i,:,:,0]*255).astype(np.uint))
            imsave(self.save_path+'images/'+str(i)+'out1.png',(sout1[i,:,:,0]*255).astype(np.uint))
            imsave(self.save_path+'images/'+str(i)+'out2.png',(sout2[i,:,:,0]*255).astype(np.uint))
        if flag:
            save_name = self.save_path+str(epoch+1)+'_auc1:'+str(auc1)+'_auc2:'+str(auc2)+'_f1:'+str(f1)+'f2:'+str(f2)+'.h5'
            print save_name
        
            if gpus>1:
                self.model.get_layer('model_1').save(save_name)
            else:
                self.model.save(save_name)
        #print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return


def load_rgb(line):

    size = (512,512)
    #rgb = Image.open(line.split(' ')[0])
    #rgb = np.array(rgb)/255.

    rgb = cv2.imread(base_path+line.split(' ')[0])
    rgb = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    #print rgb
    rgb = rgb[:,:,::-1]

    return rgb
def load_mask(line):

    size = (512,512)

    #mask = Image.open(line.split(' ')[1]).convert('L')
    #mask = np.array(mask).reshape(512,512,1)/255. 


    mask = cv2.imread(base_path+line.split(' ')[1])
    mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(np.mean(mask,axis=-1),axis=-1)

    #for tip-dataset the forgery is 0 so change it
    #np.round(np.mean(self.y_val,axis=-1))
    if 'NIST' in base_path:
        #trans black and white
        mask = 255-mask
    
    return mask

def hpf_locatenet():
    def conv(x,filters,kernel=3,strides=1,dilation=1):
        x = Conv2D(filters=filters,kernel_size=kernel,strides=strides,dilation_rate=dilation,padding='same',activation='relu')(x)
        #x = ELU()(x)
        return x
    
    rgb = Input(shape=(512,512,3))

    coarse_in = Conv2D(filters=3,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same')(rgb)
    coarse_in.trainable = False
    x = coarse_in

    x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1)(x)

    x1 = vgg_block(x,filters=64,pooling=True)     
    #x1 = BatchNormalization()(x1)
    x2 = vgg_block(x1,filters=128,pooling=True)
    #x2 = BatchNormalization()(x2)
    x3 = vgg_block(x2,filters=256,is_seven=True)
    #x3 = BatchNormalization()(x3)
    #x3 = vgg_block(x3,filters=256,is_seven=True)
    x = conv(x3,256,dilation=2)
    x = conv(x,256,dilation=4)
    x = conv(x,256,dilation=8)
    x = conv(x,256,dilation=16)
    #x = BatchNormalization()(x)
    x3 = x
    x3 = Concatenate(name='concat1')([x2,x3])
    x4 = det_deconv(x3,64)
    #x4 = BatchNormalization()(x4)
    x5 = vgg_block(x4,filters=64)
    #x5 = BatchNormalization()(x5)
    x5 = Concatenate(name='cocat2')([x1,x5])
    x6 = det_deconv(x5,32)
    #x6 = BatchNormalization()(x6)

    out1 = Conv2D(1,7,activation='sigmoid',padding='same',name='out1')(x6)
        
  
    
    #x = Add()([out1,rgb])

    
    #fine = feature_sum
    fine = Conv2D(filters=3,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same')(x6)
    fine.trainable = False
    x = fine
    x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1)(x)
    #fine = x6
    x1 = vgg_block(x,filters=64,pooling=True)        
    #x1 = BatchNormalization()(x1)
    x2 = vgg_block(x1,filters=128,pooling=True)
    #x2 = BatchNormalization()(x2)
    x3 = vgg_block(x2,filters=256,is_seven=True)
    #x3 = BatchNormalization()(x3)
    #x3 = vgg_block(x3,filters=256,is_seven=True)

 
    x = conv(x3,256,dilation=2)
    x = conv(x,256,dilation=4)
    x = conv(x,256,dilation=8)
    x = conv(x,256,dilation=16)

    pam = PAM()(x)

    pam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    #pam = BatchNormalization()(pam)

    cam = CAM()(x)
    #cam = conv(cam,768)
    #cam = Dropout(0.5)(cam)
    cam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    #cam = BatchNormalization()(cam)
    
    feature_sum = Add()([pam, cam])
    #feature_sum = Dropout(0.5)(feature_sum)
    feature_sum = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(feature_sum)
    #feature_sum = BatchNormalization()(feature_sum)
    
    x3 = feature_sum#feature_sum#vgg_block(feature_sum,filters=128)
    #x3 = pam
    x3 = Concatenate(name='concat3')([x2,x3])
    x4 = det_deconv(x3,64)  
    #x4 = BatchNormalization()(x4)
    x5 = vgg_block(x4,filters=64)
    #x5 = BatchNormalization()(x5)
    x5 = Concatenate(name='cocat4')([x1,x5])
    x6 = det_deconv(x5,32)
    #x6 = BatchNormalization()(x6)
    

    out2 = Conv2D(1,7,padding='same',activation='sigmoid',name='out2')(x6)  
    


    model = Model(inputs=[rgb], outputs=[out1,out2])
    
    if gpus>1:
        
        model = multi_gpu_model(model,gpus=gpus)
       
    #print (model.summary())
    optimizer = Adam(lr=0.0002,clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=[f1])
    
    return model


def cwhpf_locatenet():
    def conv(x,filters,kernel=3,strides=1,dilation=1):
        x = Conv2D(filters=filters,kernel_size=kernel,strides=strides,dilation_rate=dilation,padding='same',activation='relu')(x)
        #x = ELU()(x)
        return x
    
    rgb = Input(shape=(512,512,3))

    coarse_in = DepthwiseConv2D(depth_multiplier=3,kernel_size=(5,5),padding='same',depthwise_initializer=srm_init)(rgb)
    coarse_in.trainable = False
    x = coarse_in
    x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1)(x)

    x1 = vgg_block(x,filters=64,pooling=True)     
    #x1 = BatchNormalization()(x1)
    x2 = vgg_block(x1,filters=128,pooling=True)
    #x2 = BatchNormalization()(x2)
    x3 = vgg_block(x2,filters=256,is_seven=True)
    #x3 = BatchNormalization()(x3)
    #x3 = vgg_block(x3,filters=256,is_seven=True)
    x = conv(x3,256,dilation=2)
    x = conv(x,256,dilation=4)
    x = conv(x,256,dilation=8)
    x = conv(x,256,dilation=16)
    #x = BatchNormalization()(x)
    x3 = x
    x3 = Concatenate(name='concat1')([x2,x3])
    x4 = det_deconv(x3,64)
    #x4 = BatchNormalization()(x4)
    x5 = vgg_block(x4,filters=64)
    #x5 = BatchNormalization()(x5)
    x5 = Concatenate(name='cocat2')([x1,x5])
    x6 = det_deconv(x5,32)
    #x6 = BatchNormalization()(x6)

    out1 = Conv2D(1,7,activation='sigmoid',padding='same',name='out1')(x6)
        
  
    
    #x = Add()([out1,rgb])

    
    #fine = feature_sum
    fine = DepthwiseConv2D(depth_multiplier=3,kernel_size=(5,5),padding='same',depthwise_initializer=srm_init)(x6)
    fine.trainable = False
    x = fine
    x = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1)(x)
    #fine = x6
    x1 = vgg_block(x,filters=64,pooling=True)        
    #x1 = BatchNormalization()(x1)
    x2 = vgg_block(x1,filters=128,pooling=True)
    #x2 = BatchNormalization()(x2)
    x3 = vgg_block(x2,filters=256,is_seven=True)
    #x3 = BatchNormalization()(x3)
    #x3 = vgg_block(x3,filters=256,is_seven=True)

 
    x = conv(x3,256,dilation=2)
    x = conv(x,256,dilation=4)
    x = conv(x,256,dilation=8)
    x = conv(x,256,dilation=16) 
    
    pam = PAM()(x)

    pam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

    cam = CAM()(x)
    #cam = conv(cam,768)
    #cam = Dropout(0.5)(cam)
    cam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    #cam = BatchNormalization()(cam)
    
    feature_sum = Add()([pam, cam])
    #feature_sum = Dropout(0.5)(feature_sum)
    feature_sum = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(feature_sum)
    #feature_sum = BatchNormalization()(feature_sum)
    
    x3 = feature_sum#feature_sum#vgg_block(feature_sum,filters=128)
    #x3 = pam
    x3 = Concatenate(name='concat3')([x2,x3])
    x4 = det_deconv(x3,64)  
    #x4 = BatchNormalization()(x4)
    x5 = vgg_block(x4,filters=64)
    #x5 = BatchNormalization()(x5)
    x5 = Concatenate(name='cocat4')([x1,x5])
    x6 = det_deconv(x5,32)
    #x6 = BatchNormalization()(x6)
    

    out2 = Conv2D(1,7,padding='same',activation='sigmoid',name='out2')(x6)  
    


    model = Model(inputs=[rgb], outputs=[out1,out2])
    
    if gpus>1:
        
        model = multi_gpu_model(model,gpus=gpus)
       
    #print (model.summary())
    optimizer = Adam(lr=0.0002,clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=[f1])
    
    return model

def impose_heatmap_on_forged_image(heatmap, forgery):
    heatmap = Image.fromarray(heatmap).convert('RGBA')
    forgery = Image.fromarray(forgery).convert('RGBA')

    comp = Image.blend(forgery,heatmap,0.7)
    return


import matplotlib.pyplot as plt
from utils import to_sigmoid

path = 'PATH_TO_THE_FORGERY'

forgery = load_img(path)

### This code is implemented based on Tensorflow and Kears toolboxes.
# 1. load the pretrained models
hpf_model = hpf_locatenet()
hpf_model.load_weights('hpf_locatenet.h5')
cwhpf_model = cwhpf_locatenet()
cwhpf_model.load_weights('cwhpf_locatenet.h5')
# 2. extract the intermediate feature maps of the second blocks of HPF or CW-HPF 
layer_name = 'hpf_refine_block'
hpf_intermediate_layer_model = Model(inputs=hpf_model.input,
                                 outputs=hpf_model.get_layer(layer_name).output)
hpf_intermediate_output = hpf_intermediate_layer_model.predict(forgery)

cwlayer_name = 'cwhpf_refine_block'
cwhpf_intermediate_layer_model = Model(inputs=cwhpf_model.input,
                                 outputs=cwhpf_model.get_layer(cwlayer_name).output)
cwhpf_intermediate_output = cwhpf_intermediate_layer_model.predict(forgery)
# 3. calculate the heatmap.
hpf_heatmap = to_sigmoid(hpf_intermediate_output.mean(-1))
cwhpf_heatmap = to_sigmoid(cwhpf_intermediate_output.mean(-1))

# 4. !!!enhancement for better visualization!!!
hpf_heatmap = hpf_heatmap.clip(0.4,0.52)
cwhpf_heatmap = cwhpf_heatmap.clip(0.4,0.52)

# 5. visualization
plt.imshow(hpf_heatmap,cmap='bwr')  
plt.imshow(cwhpf_heatmap,cmap='bwr')  
impose_heatmap_on_forged_image(hpf_heatmap,forgery)
impose_heatmap_on_forged_image(cwhpf_heatmap,forgery)



forgery_attention_locatenet = cwhpf_locatenet

orginal_img_path = 'PATH_TO_Original_IMAGE'

original_img = load_img(orginal_img_path)
forgery_attention_model = forgery_attention_locatenet()
forgery_attention_model.load_weights('forgery_attention_locatenet.h5')

forgery_attention_output_name = 'forgery_attention_output'
fa_intermediate_layer_model = Model(inputs=forgery_attention_model.input,
                                 outputs=forgery_attention_model.get_layer(forgery_attention_output_name).output)
forgery_attention_intermediate_output = fa_intermediate_layer_model.predict(original_img)

forgery_attention_heatmap = to_sigmoid(forgery_attention_intermediate_output.mean(-1))

forgery_attention_heatmap = forgery_attention_heatmap.clip(0.38,0.55)

plt.imshow(forgery_attention_heatmap,cmap='bwr')  
impose_heatmap_on_forged_image(forgery_attention_heatmap,original_img)
