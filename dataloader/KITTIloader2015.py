import os
import os.path


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def kittidataloader(filepath, split=160):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'
  disp_R = 'disp_occ_1/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  train = image[:split]
  val   = image[split:]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]

  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
