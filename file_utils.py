from PIL import Image
import scipy.io
import numpy as np
import math

def pad_for_division_n(arr, n=16):
  """
  Pads a numpy array with zeros to make it divisible by 2 n times. 
  Params:
    arr: Array to apply padding
    n: Times to divide the array by 2 
  Returns padded array
  """
  if len(arr.shape) == 3:
    h,w,c = arr.shape

  else:
    h,w = arr.shape
  
  h_new = math.ceil( h / n) * n
  w_new = math.ceil( w / n) * n

  if len(arr.shape)== 3:
    x_new = np.zeros((h_new, w_new, c))
    x_new[:h, :w, :c] = arr.copy()
  else:
    x_new = np.zeros((h_new, w_new))
    x_new[:h, :w] = arr.copy()
    
  return x_new

def load_mask(mask_path, preprocess=None, target_size=None):
  """
  Loads mask file from given file path.
  Params:
    mask_path: Path of the mask to be loaded.
  Returns:
    Loaded mask file    
  """
  loaded_cls = scipy.io.loadmat(mask_path)
  np_mask = loaded_cls['GTcls'][0][0][1]
  if target_size is not None:
    
    np_mask = Image.fromarray(np_mask)
    np_mask = np_mask.resize(target_size, Image.NEAREST)
    np_mask = np.array(np_mask)
  
  if preprocess is not None:
      return preprocess(np_mask)
  return np_mask

def load_img(img_path, preprocess=None, target_size=None):
  """
  Loads image file in given target size and applies preprocess to loaded file.
  Params:
    img_path: Path of the image file to be loaded.
    preprocess: Preprocess function to apply the images. If passed as None then
    no preprocess will be applied. 
    target_size: Size of the image to be loaded in (width, height). If passed 
    as None then the image will be loaded in its orginal size
  Returns:
    Preprocess applied loaded image in target size 
  """
  img = Image.open(img_path)
  
  if target_size is not None:
    img = img.resize(target_size)
  img = np.array(img)
  
  if preprocess is not None:
    return preprocess(img)

  return img

def mask_name_to_img_name(mask_name):
  """
  Converts given mask name to image name. Used for mapping from mask name to 
  image name
  Params:
    mask_name: Name of the mask name
  Returns:
    Name of the image file mapped from mask name
  """
  return mask_name.replace('.mat', '.jpg')