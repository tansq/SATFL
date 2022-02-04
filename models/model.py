import tensorflow as tf
import  keras

from keras.layers import Conv2D, UpSampling2D, InputLayer, Lambda , Input , Reshape , Activation, Dense, Dropout, Flatten,ELU
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D,AveragePooling2D,UpSampling2D,SeparableConv2D,LeakyReLU,Concatenate,DepthwiseConv2D,Add
from keras.optimizers import SGD,Adam


from keras.utils import multi_gpu_model
from .modules import vgg_block,det_deconv,CAM,PAM,srm_init




def locatenet(gpus=1):
    def conv(x,filters,kernel=3,strides=1,dilation=1):
        x = Conv2D(filters=filters,kernel_size=kernel,strides=strides,dilation_rate=dilation,padding='same',activation='relu')(x)
        #x = ELU()(x)
        return x
    
    rgb = Input(shape=(None,None,3))

    coarse_in = DepthwiseConv2D(depth_multiplier=3,kernel_size=(5,5),padding='same',depthwise_initializer=srm_init)(rgb)
    coarse_in.trainable = False
    x = coarse_in
    #x = Conv2D(filters=3,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same',input_shape=(256,256,1))(rgb)
    #x = Conv2D(filters=32)

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
    
    #fine = x6
    x1 = vgg_block(fine,filters=64,pooling=True)        
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
    cam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    
    feature_sum = Add()([pam, cam])
    feature_sum = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(feature_sum)
    
    x3 = feature_sum
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
                      optimizer=optimizer)
    
    return model

    