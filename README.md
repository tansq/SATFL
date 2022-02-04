# **Self-Adversarial Training incorporating Forgery Attention for Image Forgery Localization**
## This is the official repo for our paper. The source codes and a demo are provided in this repo.

> Dirs:<br>
Models-> Source codes of our framework<br>
checkpoints-> The pretrained model fine-tuned on NIST dataset<br>
samples-> Several forged samples and the corresponding masks


## 1. Dependency
Install the relevant libs according to requirements.txt. Our codes are written in Keras with the TensorFlow backend.
```
tensorflow==1.14.0
keras==2.3.1

```
## 2. Demo
View and run demo.ipynb with Jupyter Notebook. Some examples and demo are provided in demo.ipynb. One may view some samples predicted by our model and run this demo to localize the tampered regions of forged image.

> Kindly note that the pretrained model was only fine-funed on NIST dataset. Thus, to obtain more robust performance, one need to train the model on more datasets. 

## 3. Retrain

Modify train.py and load_dataset.py to load new datasets. Run the code for training: 

```
python train.py
```

## 4. Visualization

!!!Updated for visulization!!!

To view the feature maps of the CW-HPF and Forgery Attention Module, we provide the code for visulization. Please run the code as follows:

'''
python models/vis_code.py
'''


