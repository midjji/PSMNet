import torch.utils.data as data
import random
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch



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
        transforms.Normalize(**imagenet_stats),
    ]
    tr=transforms.Compose(t_list)
    tst=tr(img)
    return tst

def default_loader(path):
    cv2im=cv2.imread(path)
    cv2im=cv2.cvtColor(cv2im,cv2.COLOR_BGR2RGB)

    return preprocess(cv2im)#,(Image.open(path).convert('RGB')) # same as pilim after postprocess...



def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
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
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img[:, y1:y1 + th, x1:x1 + tw]  # correct preprocessing...
           right_img = right_img[:, y1:y1 + th, x1:x1 + tw]  # correct preprocessing...
           dataL = dataL[y1:y1 + th, x1:x1 + tw]/256 # why divide by 256?
           print(dataL)


           return left_img, right_img, dataL
        else:
            max_rows, max_cols = (376, 1242)
            buff = np.zeros(376, 1242, 3)
            buff[0:left_img.shape[0], left_img.shape[1], :] = left_img


           w, h = left_img.size
            # again changing the size? why!!!!
           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1232, h-368, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
