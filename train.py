from models import localnet,load_rgb,load_mask,imageLoader,flip,rotation,f1_score,fgsm,load_dataset_files
import os
from keras.callbacks import Callback
import concurrent.futures as cf
import random
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = 1
batch_size = gpus*2

epoch = 100
dataset = 'nist'

train_files,valid_files = load_dataset_files(dataset)


if __name__=='__main__':

    sat = False
    model_save = 'checkpoints/'
    
    if not os.path.exists(model_save+'images/'):
        os.makedirs(model_save+'images/')
    model = locatenet()
    model.get_layer('model_1').load_weights('checkpoints/pretrained_model.h5') 
    print ('load_weights success')

    batches = int(len(train_files)/batch_size)
    print ('success in loading valid_files....batches:',batches,len(train_files),batch_size)

    model_name = ''
    
    y_pred = model.output[1]
    grads_logits = K.gradients(y_pred, model.input)[0]
    _grads = K.function([model.input], [grads_logits])
    
    for e in range(epoch):

        random.shuffle(train_files)
        
        
        print 'Epoch ',e,'/',epoch
        
        for batch in range(batches):
            files = train_files[batch*batch_size:(batch+1)*batch_size]
            
            with cf.ThreadPoolExecutor() as executor:
                
                rgb = executor.map(load_rgb,files)
                mask = executor.map(load_mask,files)
                rgb = list(rgb)
                mask = list(mask)
                
                X = np.array(rgb)/255.
     
                Y = (np.array(mask)/255.).astype(np.uint)

            logs = model.train_on_batch(X,[Y,Y])
            print ('Batch: ',batch,'/',batches,'loss:%f, out1_loss:%f, model_2_loss:%f'%(logs[0],logs[1],logs[2]))


            if sat:
                eps = random.uniform(0,0.01)#
                adv_X = fgsm(_grads,X,Y,eps)
                if not np.isnan(adv_X[-1]).any():
                    logs = model.train_on_batch(adv_X,[Y,Y])
                    print ('SAT-eps:%f: loss:%f, out1_loss:%f, model_2_loss:%f'%(eps,logs[0],logs[1],logs[2]))
            

        model_name = 'epoch-'+str(e)+'.h5'
        model.save(model_save+model_name)

