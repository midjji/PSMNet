# correct paths, if started with this one, its here
import sys
# if started from
sys.path.insert(0, "../../")
from unpackage.util import show_batch,show_tensor,show_image

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
parser.add_argument('--dataset', default='')
parser.add_argument('--test', default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


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

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))



def train(imgL, imgR, disp_L, show_images=False):
    imgL = imgL
    imgR = imgR
    disp_L = disp_L

    assert args.cuda

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    # with our datareader impossible disparities are negative instead
    mask = ((disp_true < args.maxdisp) * (disp_true >= 0))

    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)

        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask]) + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[
            mask]) + F.smooth_l1_loss(output3[mask], disp_true[mask])
        output = torch.squeeze(output2, 1)
    elif args.model == 'basic':
        output3 = model(imgL, imgR)
        output = torch.squeeze(output3, 1)
        loss = F.smooth_l1_loss(output3[mask], disp_true[mask])

    if show_images:
        show_batch(imgL, imgR, disp_true, output.detach(), 1,maxdisp=192)

    loss.backward()
    optimizer.step()
    return loss.data.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    if args.cuda:
        imgL, imgR,disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    # ---------
    mask = (disp_true < args.maxdisp) * (disp_true >= 0) ==1
    # ----

    with torch.no_grad():
        output3 = model(imgL, imgR)

    #output = torch.squeeze(output3.data.cpu(), 1)[:, 4:, :] # not needed when using our loader, the padding is

    if len(disp_true[mask]) == 0:
        loss = 0
    else:

        loss = torch.mean(torch.abs(output3[mask] - disp_true[mask]))  # end-point-error

    return loss


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    if args.loadmodel is not None:
        lr=0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

from datasetmanager import DatasetManager, DataRequirements,search_for_dataset,DisparityAugmenter
from unpackage.data_preprocessing import DisparityDatasetWrapper
from torch.utils.data import DataLoader, RandomSampler
def main():
    data_requirements = DataRequirements(include_names=["FlyingThings3D", "Driving", "Monkaa"])

    dataset_basepath = search_for_dataset("/archive/datasets")
    dataset_manager = DatasetManager(data_requirements,
                                     basepath=dataset_basepath,
                                     cache_in_ram=False)
    dtrain,_,_ = dataset_manager.get_training_validation_testing_datasets()

    augmenter=DisparityAugmenter()
    parameters = {'subsample_input': 1,
                  'n_depths': 192,}

    dtrain=DataLoader(DisparityDatasetWrapper(dtrain,parameters), num_workers=8, batch_size=6)





    if not args.test and False:
        model.train()

        start_full_time = time.time()
        for epoch in range(1, args.epochs + 1):
            print('This is %d-th epoch' % (epoch))
            total_train_loss = 0
            adjust_learning_rate(optimizer, epoch)

            ## training ##
            for batch_idx, (left,right, disp) in enumerate(dtrain):
                start_time = time.time()
                left,right,disp=augmenter(left,right,disp)



                loss = train(left,right,disp, batch_idx % 2 == 0 and False)
                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
                total_train_loss += loss
            print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dtrain)))

            # SAVE
            savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
            }, savefilename)

        print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    savefilename = args.savemodel + 'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss / len(TestImgLoader),
    }, savefilename)


if __name__ == '__main__':
    main()

