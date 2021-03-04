import os
import numpy as np
import math
from tensorflow.keras.utils import Sequence, to_categorical
from file_utils import load_mask, load_img, mask_name_to_img_name, pad_for_division_n

mask_data_dir = 'data/benchmark_RELEASE/dataset/cls'
img_data_dir = 'data/benchmark_RELEASE/dataset/img'

tr_paths_txt = 'data/benchmark_RELEASE/dataset/train.txt'
val_paths_txt = 'data/benchmark_RELEASE/dataset/val.txt'

class DataLoader(Sequence):
  
  def __init__(self, mask_paths, image_paths, semi_mask_paths, semi_image_paths,
               num_class, batch_size, img_preprocess, mask_preprocess, target_size):
    
    self.batch_size = batch_size
    self.i_preprocess = img_preprocess
    self.m_preprocess = mask_preprocess
    self.target_size = target_size
    self.n_class = num_class

    self.mask_paths = mask_paths
    self.image_paths = image_paths
    self.semi_mask_paths = semi_mask_paths
    self.semi_image_paths = semi_image_paths
    self.len_semi = math.ceil(len(self.semi_image_paths) / self.batch_size)
    
  def __len__(self):
    return math.ceil(len(self.mask_paths) / self.batch_size)
  
  def __getitem__(self, index):

    batch_mask_paths = self.mask_paths[index*self.batch_size:(index+1)*self.batch_size]
    batch_image_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]

    if len(self.semi_image_paths) > 0:
      # Select examples from semi randomly
      idx = np.arange(len(self.semi_mask_paths))
      # Take same number of semi examples with the fully training examples to prevent problems in batch update
      selected_idx = np.random.choice(idx, size=len(batch_image_paths), replace=False)

      batch_semi_mask_paths = np.array(self.semi_mask_paths)[selected_idx].tolist()
      batch_semi_image_paths = np.array(self.semi_image_paths)[selected_idx].tolist()
    

    batch_masks = np.array([load_mask(mpath, self.m_preprocess, self.target_size) for mpath in batch_mask_paths])
    batch_images = np.array([load_img(img_path, self.i_preprocess, self.target_size) for img_path in batch_image_paths])

    if len(self.semi_image_paths) > 0:
      batch_semi_masks = np.array([load_mask(mpath, self.m_preprocess, self.target_size) for mpath in batch_semi_mask_paths])
      batch_semi_images = np.array([load_img(img_path, self.i_preprocess, self.target_size) for img_path in batch_semi_image_paths])

    # Make the first shape 1 if it is not
    if batch_masks.shape == 3:
      batch_masks = np.expand_dims(batch_masks, axis=0)
      batch_semi_masks = np.expand_dims(batch_semi_masks, axis=0)

    if batch_images.shape == 3 and len(self.semi_image_paths) > 0:
      batch_images = np.expand_dims(batch_images, axis=0)
      batch_semi_images = np.expand_dims(batch_semi_images, axis=0)

    if len(self.semi_image_paths) > 0:
      return batch_images, to_categorical(batch_masks, num_classes=self.n_class), batch_semi_images
    
    return batch_images, to_categorical(batch_masks, num_classes=self.n_class)

def get_voc_datagen(num_class, batch_size, img_preprocess, mask_preprocess, target_size, portion_semi=0.0):

  with open(tr_paths_txt, 'r') as txtfile:
    lines = txtfile.readlines()
    train_names = [l.strip() for l in lines]

  with open(val_paths_txt, 'r') as txtfile:
    lines = txtfile.readlines()
    val_names = [l.strip() for l in lines]

  # Split some images for semi-supervised learning 
  num_semi = int(len(train_names) * portion_semi)
  semi_train_names = np.random.choice(train_names, size=num_semi, replace=False).tolist()
  train_names = [fname for fname in train_names if fname not in semi_train_names]
  
  tr_mask_paths = [os.path.join(mask_data_dir, mname + '.mat') for mname in train_names]
  tr_image_paths = [os.path.join(img_data_dir, imgname + '.jpg') for imgname in train_names]
  
  val_mask_paths = [os.path.join(mask_data_dir, mname + '.mat') for mname in val_names]
  val_image_paths = [os.path.join(img_data_dir, imgname + '.jpg') for imgname in val_names]

  semi_mask_paths = [os.path.join(mask_data_dir, mname + '.mat') for mname in semi_train_names]
  semi_image_paths = [os.path.join(img_data_dir, imgname + '.jpg') for imgname in semi_train_names]

  tr_gen = DataLoader(tr_mask_paths, tr_image_paths, semi_mask_paths, semi_image_paths, num_class, batch_size, img_preprocess, mask_preprocess, target_size)
  val_gen = DataLoader(val_mask_paths, val_image_paths, [], [], num_class, batch_size, img_preprocess, mask_preprocess, target_size)
  return tr_gen, val_gen