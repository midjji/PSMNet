import torch.utils.data as data
import random
from PIL import Image
from .readpfm import readPFM
import numpy as np
import torchvision.transforms as transforms
import cv2
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
    cv2im=cv2im/ 255 - 0.5

    return preprocess(cv2im)#,(Image.open(path).convert('RGB')) # same as pilim after postprocess...





def disparity_loader(path):

    return torch.tensor(np.ascontiguousarray(readPFM(path),dtype=np.float32))


class FreiburgLoader(data.Dataset):
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

        else:

            dim, h, w = left_img.shape
            l=torch.tensor(np.zeros((3,544,w), dtype=np.float32 ))
            l[:,4:,:]=left_img
            left_img=l
            r = torch.tensor(np.zeros((3,544,w), dtype=np.float32 ))
            r[:, 4:, :] = right_img
            right_img = r
            d = torch.tensor(np.zeros((3,544,w), dtype=np.float32 ))
            d[:, 4:, :] = dataL
            dataL = d


        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
