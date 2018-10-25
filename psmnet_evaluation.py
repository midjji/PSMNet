import os
import sys
sys.path.append( '../../')
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import time



from models import stackhourglass, basic
import cv2
import numpy as np

from unpackage.evaluation import evaluate_model,analyze_result

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/archive/datasets/psmnetdatasets/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', default='2015')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)





if args.dataset == '2015':
    args.datapath="/archive/datasets/kitti/stereo2015/training/"
    from dataloader.KITTIloader2015 import kittidataloader
    from dataloader.KITTILoader import KittiLoader
    train_left, train_right, train_disp, test_left_img, test_right_img, test_disp = kittidataloader(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        KittiLoader(train_left, train_right, train_disp,  training=False),
        batch_size=4, shuffle=False, num_workers=8, drop_last=False)
    TestImgLoader = torch.utils.data.DataLoader(
        KittiLoader(test_left_img, test_right_img, test_disp, training=False),
        batch_size=4, shuffle=False, num_workers=8, drop_last=False)


if args.dataset == '2012':
    from dataloader import KITTI_submission_loader2012 as DA

if args.dataset == '':
    from dataloader import listflowfile as lt
    from dataloader import SceneFlowLoader as DA

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
    something=[all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp]
    for l in something:
        l[:]=l[0:100]


    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, training=False),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, training=False),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def check_sizes(dl):
    for sample_idx, sample in enumerate(dl):
            print('{0:.1f}%'.format(100*sample_idx/len(dl)), end='\r')
            image_left, image_right, disp_left = sample
            if sample_idx == 0:
                GOOD_AND_PROPER_SIZE_LEFT = image_left.shape
                GOOD_AND_PROPER_SIZE_RIGHT = image_right.shape
                GOOD_AND_PROPER_SIZE_DISP = disp_left.shape

            print("Index: {}".format(sample_idx))
            assert GOOD_AND_PROPER_SIZE_DISP == disp_left.shape, "Wrong size on disp, got {}, expected {}".format(GOOD_AND_PROPER_SIZE_DISP, disp_left.shape)
            assert GOOD_AND_PROPER_SIZE_LEFT == image_left.shape, "Wrong size on disp, got {}, expected {}".format(GOOD_AND_PROPER_SIZE_LEFT, image_left.shape)
            assert GOOD_AND_PROPER_SIZE_RIGHT == image_right.shape, "Wrong size on disp, got {}, expected {}".format(GOOD_AND_PROPER_SIZE_RIGHT, image_right.shape)



def show_image(img, name=""):
    cv2.namedWindow(name,0)
    cv2.imshow(name,img)


def test_image_crop_disparity():
    for sample_idx, sample in enumerate(TrainImgLoader):
        image_left, image_right, disp_left = sample
        show_image(image_left,"image_left")
        show_image(image_right, "image_right")
        show_image(disp_left, "disp")
        cv2.waitKey(0)



def main():
    #test_image_crop_disparity()
    #test_change_shape_pil()


    model.eval()
    train_errors = analyze_result(evaluate_model(model, TrainImgLoader))
    print(train_errors)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Done evaluating Training data")


    validation_errors = analyze_result(evaluate_model(model, TestImgLoader))
    print(validation_errors)
    print("Done evaluating validation errors!")




if __name__ == '__main__':
    main()

