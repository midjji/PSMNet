import torch.utils.data as data
import random
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
from .readpfm import readPFM



def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
# this is all it does!
def preprocess(img):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    t_list = [
        transforms.ToTensor(),
       # transforms.Normalize(**imagenet_stats),
    ]
    tr=transforms.Compose(t_list)
    tst=tr(img)
    return tst

def pad2size(npimg, rows,cols):
    if len(npimg.shape)==3:
        buff=np.zeros((rows,cols,3),dtype=np.float32)
        buff[0:npimg.shape[0],0:npimg.shape[1],:] = npimg
    if len(npimg.shape) == 2:
        buff = np.zeros((rows, cols), dtype=np.float32)
        buff[0:npimg.shape[0], 0:npimg.shape[1]] = npimg

    # why would it be top left padding? bottom right makes much more sense...
    # this is what they did
    # datasize should be
    # pad to (384, 1248)
    # top_pad = 384 - cv2im.shape[0]
    # left_pad = 1248 - cv2im.shape[1]
    # cv2im = np.pad(cv2im,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    return buff

def default_loader(path):
    cv2im=cv2.imread(path)
    cv2im=cv2.cvtColor(cv2im,cv2.COLOR_BGR2RGB)
    cv2im=pad2size(cv2im,384,1248)/255 -0.5




    return preprocess(cv2im)#,(Image.open(path).convert('RGB')) # same as pilim after postprocess...



def disparity_loader(path):
    cv2im=cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    cv2im=cv2im/256 # data is stored as ushort with 256 decimal
    cv2im=pad2size(cv2im,384,1248)
    return torch.tensor(cv2im)





class KittiLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):

        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:

            dim, h, w = left_img.shape
            th, tw = 256, 512
 
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img[:, y1:y1 + th, x1:x1 + tw]  # correct preprocessing...
            right_img = right_img[:, y1:y1 + th, x1:x1 + tw]  # correct preprocessing...
            dataL = dataL[y1:y1 + th, x1:x1 + tw]
            return left_img, right_img, dataL


        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
