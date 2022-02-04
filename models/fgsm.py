
import numpy as np
       
from keras.losses import binary_crossentropy

def fgsm( image, y_true,_grads,eps=.01):
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