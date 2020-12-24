# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:20:48 2019

@author: moona
"""

#%% Fast ai codes
import numpy as np
import os
import time
import cv2
#import nibabel as nib
import pdb
from matplotlib import pyplot as plt
import nibabel as nib
from nibabel.testing import data_path

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

path_lbl = '/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/segthordataset/segthormasks/'
path_img = '/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/segthordataset/Segthorimages/'
fnames = get_image_files(path_img)
fnames[:3]

lbl_names = get_image_files(path_lbl)
lbl_names[:3]

img_f = fnames[4]
#img = open_image(img_f)
#img.show(figsize=(5,5))
import numpy as np
get_y_fn = lambda x:path_lbl+f'{(x.stem)}__label{x.suffix}'
#codes = array(['heart', 'aorta', 'esophgus', 'trachea','Void'])
#codes=array(['Glnd_Submand_L', 'Glnd_Submand_R', 'LN_Neck_II_L', 'LN_Neck_II_R','LN_Neck_III_L','LN_Neck_III_R','Parotid_L','Parotid_R','Void'])
mask = open_mask(get_y_fn(img_f))
#mask.show(figsize=(5, 5), alpha=1)
src_size = np.array(mask.shape[1:])
print(src_size)
mask.data

size = src_size
codes = array(['heart', 'aorta', 'esophgus', 'trachea','Void'])
#free = gpu_mem_get_free_no_cache()
## the max size of bs depends on the available GPU RAM
#if free > 8200: bs=8
#else:           bs=4
#print(f"using bs={bs}, have {free}MB of GPU RAM free")
bs=8

src = (SegmentationItemList.from_folder(path_img)
       .split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

def dice_coeff(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total
def iou(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
  
def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics=acc

wd=1e-2
learn = unet_learner(data, models.resnet18, metrics=metrics, wd=wd)

from torchsummary import summary
summary(learn.model, input_size=(3, 512, 512))

learn.model

lr_find(learn)
learn.recorder.plot()

lr=3e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

learn.save('stage-11-imp')

learn.load('stage-11-imp');
#learn.show_results(rows=20, figsize=(8,9))
#img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_41/Patient_41_1_image.png')
##img
#im_g=learn.predict(img)
#na1=(np.array(im_g[1][0]))
#plt.imshow(na1)
#plt.show()
import matplotlib
from matplotlib import pyplot as plt

import nibabel as nib
for i in range(33,41):
    j=0
    l=1
    img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(1)+'_image.png')
    im_g=learn.predict(img)
    na=np.array(im_g[1][0])
    for k in os.listdir('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)):
        j=j+1
    while(l<(j)):
        img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(l+1)+'_image.png')
        im_g=learn.predict(img)
        a=np.array(im_g[1][0])
        na=np.dstack((na,a))
        l=l+1
    ex="test/Patient_"+str(i)+".nii.gz"
    img = nib.load(ex)
    im1=np.array(img.get_affine())
    new_image = nib.Nifti1Image(np.asarray(na,dtype="uint8" ), affine=im1)
    nib.save(new_image,'/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/3dtestingnew1/Patient_'+str(i)+'_GT.nii.gz')
 
import nibabel as nib
import cv2
import numpy as np
import os
#result_array = np.empty((512, 512,3))
#
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
#path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testdata'
#oslist=os.listdir(path1)
##train_ids = next(os.walk(path1+"Patient"))[1]
##train_ids = os.walk(path1+"train_24")
##images=next(train_ids)
##for ii in oslist:
##    print(os.path.join(data_pathnew,ii))
##    list2=os.path.join(data_pathnew,ii)
##    for jj in list2:
##        print(jj)
##        file=os.path.join(data_pathnew,ii)
##        print(os.path.join(data_pathnew,ii))
#import natsort
#import matplotlib.pyplot as plt
#im_height=512
#im_width=512
#from tqdm import tqdm
#for i, volume in enumerate(oslist):
#    print(i)
#    print(volume)
#    cur_path = os.path.join(path1, volume)
#    files=natsort.natsorted(os.listdir(cur_path))[1]
#    files1=os.path.join(cur_path,files)
#    files11=natsort.natsorted(os.listdir(files1))
#    #train_ids = next(os.walk(path1+"images"))
#    X_train = np.zeros((len(files11),im_height, im_width), dtype=np.uint8)
#    for n, id_ in tqdm(enumerate(files11), total=len(files11)):
#        print(n)
#        print(id_)
#        img=np.load(os.path.join(files1,id_))
#        print(img.shape)
#        #img=np.swapaxes(img,0,2)
#        #img1=img.transpose(1, 2, 0)
#        #img=img[1,:,:]
#        x = np.array(img)
#        im_g=learn.predict(x)
#        na=np.array(im_g[1][0])
#        X_train[n] = na
#        ff=np.swapaxes(X_train,0,2)
#    i=i+24
#    ex="D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testaapmnib/Patient_"+str(i)+".nii.gz"    
#    img = nib.load(ex)
#    im1=np.array(img.get_affine())
#    new_image = nib.Nifti1Image(np.asarray(ff,dtype="uint8" ), affine=im1)
#    nib.save(new_image,'D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testsegthornii1mask/Patient_'+str(i)+'_GT.nii.gz')
#     
learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(12, lrs, pct_start=0.8)
learn.save('stage-2');
size = src_size

#free = gpu_mem_get_free_no_cache()
## the max size of bs depends on the available GPU RAM
#if free > 8200: bs=3
#else:           bs=1
#print(f"using bs={bs}, have {free}MB of GPU RAM free")
bs=8
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data, models.resnet50, metrics=metrics, wd=wd)

learn.load('stage-2');
lr_find(learn)
#learn.recorder.plot()
lr=1e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

import matplotlib
from matplotlib import pyplot as plt

import nibabel as nib
for i in range(33,41):
    j=0
    l=1
    img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(1)+'_image.png')
    im_g=learn.predict(img)
    na=np.array(im_g[1][0])
    for k in os.listdir('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)):
        j=j+1
    while(l<(j)):
        img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(l+1)+'_image.png')
        im_g=learn.predict(img)
        a=np.array(im_g[1][0])
        na=np.dstack((na,a))
        l=l+1
    ex="test/Patient_"+str(i)+".nii.gz"
    img = nib.load(ex)
    im1=np.array(img.get_affine())
    new_image = nib.Nifti1Image(np.asarray(na,dtype="uint8" ), affine=im1)
    nib.save(new_image,'/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/3dtestingnew11/Patient_'+str(i)+'_GT.nii.gz')
 
learn.save('stage-11-big')

learn.load('stage-11-big');
learn.unfreeze()

lrs = slice(1e-6,lr/10)
learn.fit_one_cycle(10, lrs)
import matplotlib
from matplotlib import pyplot as plt

import nibabel as nib
for i in range(33,41):
    j=0
    l=1
    img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(1)+'_image.png')
    im_g=learn.predict(img)
    na=np.array(im_g[1][0])
    for k in os.listdir('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)):
        j=j+1
    while(l<(j)):
        img = open_image('/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/testing_2d/Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(l+1)+'_image.png')
        im_g=learn.predict(img)
        a=np.array(im_g[1][0])
        na=np.dstack((na,a))
        l=l+1
    ex="test/Patient_"+str(i)+".nii.gz"
    img = nib.load(ex)
    im1=np.array(img.get_affine())
    new_image = nib.Nifti1Image(np.asarray(na,dtype="uint8" ), affine=im1)
    nib.save(new_image,'/raid/Home/Users/aqayyum/TestMulticlass/fastaicodes/3dtestingnew1111/Patient_'+str(i)+'_GT.nii.gz')

learn.save('stage-2-big')
print('done all configurations')
#learn.load('stage-2-big');
#learn.show_results(rows=3, figsize=(10,10))
#nii='Patient_41_GT.nii.gz'
#import nilearn
#from nilearn import plotting
#plotting.plot_stat_map(nii)
#nii='GT.nii.gz'
#import nilearn
#from nilearn import plotting
#plotting.plot_stat_map(nii)