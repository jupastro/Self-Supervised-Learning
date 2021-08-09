import numpy as np
import os
import random
import tensorflow as tf
import cv2
from skimage.util import img_as_ubyte
from skimage import io,color
import matplotlib.pyplot as plt

def create_patches( imgs,patch_size,add_noise=False,noise_level=0):
    ''' Create a list of  patches out of a list of images
    Args:
        imgs: list of input images
        patch_size:list including both dimensions (256,256)
        add_noise: boolean to add noise to the cropped image(useful for denoising previous steps or superresolution)
        noise_level: int between 0-255 representing the sd of the gaussian noise added
    ¡¡¡¡¡IMPORTANT if the image is not in a greyscale of 0-255 the noise must be rescaled in between 0-1 !!!!
        percentage_data:0-1 float specifying the percentage of data used for training
    Returns:
        list of image patches
    '''
    
    
    patches = [] #empty list to store the corresponding patches 
    patch_height=patch_size[0]
    patch_width=patch_size[1]
    for n in range( 0, len( imgs ) ):
        image = imgs[ n ]
        original_size = imgs[n].shape
        num_y_patches = original_size[ 0 ] // patch_size[0]#obtain the int number of patches that can be actually extracted from the original image
        num_x_patches = original_size[ 1 ] // patch_size[1]
        for i in range( 0, num_y_patches ):
            for j in range( 0, num_x_patches ):
              if add_noise:
                trainNoise = np.random.normal(loc=0, scale=noise_level, size=(patch_width,patch_height))
                patches.append(np.clip(image[ i * patch_width : (i+1) * patch_width,
                                      j * patch_height : (j+1) * patch_height ]+trainNoise,0,255)  )
              else:
                patches.append(image[ i * patch_width : (i+1) * patch_width,
                                      j * patch_height : (j+1) * patch_height ]  )
    
    return patches
def set_seed(seedValue=42):
  """Sets the seed on multiple python modules to obtain results as
  reproducible as possible.
  Args:
  seedValue (int, optional): seed value.
  """
  random.seed(a=seedValue)
  np.random.seed(seed=seedValue)
  tf.random.set_seed(seedValue)
  os.environ["PYTHONHASHSEED"]=str(seedValue)
  
def shuffle_fragments( imgs,number_of_patches=(3,3)):
    ''' Shuffles different fragments of the input imgs
    Args:
        imgs: list of input images
        number_of_patches: (x,y) containing the number of divisions per x and per y
    Returns:
        list of image patches
    '''
    patches=[]
    
    original_size = imgs.shape
    img=1*imgs# This multiplication is made to avoid further relating both variables
    num_y_patches = number_of_patches[1]#obtain the int number of patches that can be actually extracted from the original image
    num_x_patches = number_of_patches[0]
    patch_height=original_size[0]//num_x_patches
    patch_width=original_size[1]//num_y_patches
    for i in range( 0, num_y_patches ):
                for j in range( 0, num_x_patches ):
                 
                    patches.append(img[ i * patch_width : (i+1) * patch_width,
                                          j * patch_height : (j+1) * patch_height ]  )
    k=0
    random.shuffle(patches)
    for i in range( 0, num_y_patches ):
            for j in range( 0, num_x_patches ):
              
                img[ i * patch_width : (i+1) * patch_width,
                                          j * patch_height : (j+1) * patch_height ]=patches[k]
                k+=1
    return img

def hide_fragments( imgs,patch_size,percent):
    ''' Sets to 0 different fragments of the input imgs
    Args:
        imgs: list of input images
        patch_size: list including both dimensions of the fragment to  (256,256)
        percent: representing the percentage of the total image to set to 0
    Returns:
        list of image patches
    '''
    patch_height=patch_size[0]
    patch_width=patch_size[1]
    original_size = imgs.shape
    img=1*imgs# This multiplication is made to avoid further relating both variables
    num_y_patches = original_size[ 0 ] // patch_size[0]#obtain the int number of patches that can be actually extracted from the original image
    num_x_patches = original_size[ 1 ] // patch_size[1]
    n=percent*num_y_patches*num_x_patches

    for w in range(0,int(n)):
        i=random.choice(range( 0, num_y_patches ))
        j=random.choice(range( 0, num_x_patches ))
        img[ i * patch_width : (i+1) * patch_width,
                                  j * patch_height : (j+1) * patch_height ]=0
    
    return img


def add_Gaussian_Noise(image,percentage_of_noise,print_img=False):
  """
  image:  image to be added Gaussian Noise with 0 mean and a certain std
  percentage_of_noise:similar to 1/SNR, it represents the % of 
  the maximum value of the image that will be used as the std of the Gaussian Noise distribution
  """
  max_value=np.max(image)
  noise_level=percentage_of_noise*max_value
  Noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
  noisy_img=np.clip(image+Noise,0,max_value)  
  if print_img:
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow( image, 'gray' )
    plt.title( 'Original image' );
    # and its "ground truth"
    plt.subplot(1, 2, 2)
    plt.imshow( noisy_img, 'gray' )
    plt.title( 'Noisy image' );
  
  return noisy_img

def crappify(img,resizing_factor,add_noise=True,noise_level=None,Down_up=True):
 
  """
  img: img to be modified
  resizing_factor(float): downsizing factor to divide the number of pixels with
  add_noise(boolean): indicating whether to add gaussian noise before applying the resizing 
  noise_level(float): number between ]0,1] indicating the std of the Gaussian noise N(0,std)
  Down_up(boolean): indicating whether to perform a final upsampling operation 
  to obtain an image of the same size as the original but with the corresponding loss of quality of downsizing and upsizing
  """
  w,h=img.shape
  org_sz=(w,h)
  new_w=int(w/np.sqrt(resizing_factor))
  new_h=int(h/np.sqrt(resizing_factor))
  targ_sz=(new_w,new_h)
  #add Gaussian noise
  if add_noise:
    noisy=add_Gaussian_Noise(img,noise_level,print_img=False)
    #downsize_resolution
    resized = cv2.resize(noisy, targ_sz, interpolation = cv2.INTER_LINEAR)
    #upsize_resolution
    if Down_up:
      resized=cv2.resize(resized, org_sz, interpolation = cv2.INTER_LINEAR)
  else:
    #downsize_resolution
    resized = cv2.resize(img, targ_sz, interpolation = cv2.INTER_LINEAR)
    #upsize_resolution
    if Down_up:
      resized=cv2.resize(resized, org_sz, interpolation = cv2.INTER_LINEAR)

  return resized

def reduce_number_imgs(imgs,label_imgs,percentage_data=1,normalize=True,imagenet=False):
    """
    Input:
    imgs:a list or tensor containing several images to be packed as a list after reducing its number
    label_imgs: a list or tensor containing several label images in the same order as the imgs tensor
    percentage_data: float(0-1) indicating the reduction in labels to be performed i.e 1 means that all the image will be taken into account
    normalize: Boolean indicating whether or not to perform a normalization step in the img, no normalization is performed in the labels as it is supposed that they would already been in a binary 
    Output:
    x: list containing a subset of imgs
    y:list containing a subset of labels
    """
    n=len(imgs)
    if imagenet:
      if normalize:
        
        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [cv2.normalize(imgs[i]/np.max(imgs[i]), None, 0, 1, cv2.NORM_MINMAX) for i in idx] 
        y= [label_imgs[i] for i in idx] 
      else:
        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [color.gray2rgb(imgs[i]) for i in idx] 
        y= [label_imgs[i] for i in idx] 
    else:
      if normalize:
        
        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [cv2.normalize(imgs[i]/np.max(imgs[i]), None, 0, 1, cv2.NORM_MINMAX) for i in idx] 
        y= [label_imgs[i] for i in idx] 
      else:
        idx=random.sample(list(range(0,n)),int(n*percentage_data))
        x= [imgs[i] for i in idx] 
        y= [label_imgs[i] for i in idx] 
    print('Created list with '+str(len(x))+' images')
   
    return x,y

def append_blackborder(img,height,width):
  """ Function to append a blackborder to the images in order to avoid a resizing step that may affect the resolution and pixel size
  """
  new_h=(height-img.shape[0])
  new_w=(width- img.shape[1])
  img = cv2.copyMakeBorder(img ,new_h,0,new_w,0 , cv2.BORDER_CONSTANT) 
  return img
def append_pot2(img):
  """
  Function to append a blackborder but instead of having to specify the shape of the desired image
  the function would check the shape and append a black border in order to obtain an image that is a multiple of 2^n as required by the U-Net Models
  
  """
  new_height=img.shape[0]
  new_width=img.shape[1]
  while new_height%32!=0:
    new_height+=1
  while new_width%32!=0:
    new_width+=1
  img = append_blackborder(img,new_height,new_width)
  #print('An image with shape'+str(img.shape)+'has been created')
  return img
