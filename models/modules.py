import tensorflow as tf
import  keras
import numpy as np
import glob
from keras.layers import Conv2D, UpSampling2D, InputLayer, Lambda , Input , Reshape , Activation, Dense, Dropout, Flatten,ELU
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D,AveragePooling2D,UpSampling2D,SeparableConv2D,LeakyReLU,Concatenate,DepthwiseConv2D,Add
from keras import backend as K
from keras.layers import Layer
import keras.layers as layers

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
