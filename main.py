
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import time

from dataloader import listflowfile as lt
from dataloader import SceneFlowLoader as DA
from models import stackhourglass,basic
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/archive/datasets/psmnetdatasets/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
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
    train_left, train_right, train_disp, test_left_img, test_right_img, test_disp = kittidataloader(args.datapath, split=160)
    TrainImgLoader = torch.utils.data.DataLoader(
        KittiLoader(train_left, train_right, train_disp,  training=True),
        batch_size=5, shuffle=False, num_workers=8, drop_last=False)
    TestImgLoader = torch.utils.data.DataLoader(
        KittiLoader(test_left_img, test_right_img, test_disp, training=True),
        batch_size=5, shuffle=False, num_workers=8, drop_last=False)




if args.dataset == '':
    from dataloader import listflowfile as lt
    from dataloader import SceneFlowLoader as DA

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
    something=[all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp]
    for l in something:
        l[:]=l[0:100]


    TrainImgLoader = torch.utils.data.DataLoader(
        DA.FreiburgLoader(all_left_img, all_right_img, all_left_disp, training=True),
        batch_size=5, shuffle=False, num_workers=0, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.FreiburgLoader(test_left_img, test_right_img, test_left_disp, training=True),
        batch_size=5, shuffle=False, num_workers=0, drop_last=False)






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



def show_image(rgb, name=""):

    if(len(rgb.shape)==3):
        img=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    else:
        img=rgb

    cv2.namedWindow(name,0)
    cv2.imshow(name,img)

def tensor2rgb(tensor):
    # base transform is to -0.5 to 0.5
    if(len(tensor.shape)==3):
        return np.transpose(tensor.cpu().numpy(),(1,2,0)) + 0.5
    return tensor.cpu().numpy().astype(np.float32)


def show_tensor(tensor, name=""):
    show_image(tensor2rgb(tensor),name)


def show_batch(left,right,disp,estimate,max=1):
    mask = ((disp < args.maxdisp) * (disp > 0))
    for b in range(min(max,left.shape[0])):


        show_tensor(left[b,:,:,:].squeeze(), "left: "+str(b))
        show_tensor(right[b, :, :,:].squeeze(), "right: " + str(b))
        show_tensor(disp[b, :, :].squeeze(), "disp: " + str(b))
        show_tensor(estimate[b, :, :].squeeze(), "estimate: " + str(b))
        show_tensor(mask[b,:,:].squeeze(),"mask")


        diff = torch.abs(estimate[b,:,:] - disp[b,:,:]) * mask[b,:,:].float()

        show_tensor(diff,"diff")
        #diff=diff>=3
        #show_tensor(diff,"errors")



    cv2.waitKey(110)








def train(imgL,imgR, disp_L, show_images=False):

        imgL   = imgL
        imgR   = imgR
        disp_L = disp_L


        assert args.cuda

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

       #---------
        # must be >0 because 0 is their flag for failure... that collides with usual use of negative numbers...
        mask =((disp_true < args.maxdisp ) * (disp_true >0))



        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask]) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask]) + F.smooth_l1_loss(output3[mask], disp_true[mask])
            output = torch.squeeze(output2, 1)
        elif args.model == 'basic':
            output3 = model(imgL,imgR)
            output = torch.squeeze(output3,1)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask])

        if show_images:
                show_batch(imgL,imgR,disp_true,output.detach(), max=1)
                cpu_mask = mask.cpu().numpy()[0,:,:]

                tmp = imgL.cpu().numpy()

                tmp = tmp[0, :, :, :]
                tmp = (np.transpose(tmp, (1, 2, 0)))

                cv2.namedWindow("imgl", 0)
                cv2.imshow("imgl", tmp)

                batchres = output.detach().cpu().numpy()

                batchres = np.maximum(0, batchres[0, :, :])
                disp_gt = disp_true.cpu().numpy()
                disp_gt = disp_gt[0, :, :]
                maxv = np.max(disp_gt)

                cv2.namedWindow("disp1", 0)

                cv2.imshow("disp1", batchres / maxv)

                cv2.namedWindow("disp gt", 0)
                cv2.imshow("disp gt", disp_gt / maxv)
                cv2.namedWindow("mask", 0)
                cv2.imshow("mask",cpu_mask.astype('float32'))

                diff = np.absolute(batchres - disp_gt) * cpu_mask

                diff1 = diff - np.min(diff)
                diff1 = diff / max(64, np.max(diff))
                diff2 = np.minimum(3, diff) // 3

                cv2.namedWindow("diff1", 0)
                cv2.imshow("diff1", diff1)
                cv2.namedWindow("diff2", 0)
                cv2.imshow("diff2", diff2)

                #computing 3-px error#
                error = np.sum((np.abs(disp_gt - batchres) >= 3) * cpu_mask)

                assert np.sum(cpu_mask== 0) + np.sum(cpu_mask == 1) == cpu_mask.size
                print('3-pixel error: {0:.1f}%'.format(100*error/np.sum(cpu_mask)))



                cv2.waitKey(100)


        loss.backward()
        optimizer.step()
        return loss.data.item()

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #---------
        mask = disp_true < args.maxdisp and disp_true >= 0
        #----

        with torch.no_grad():
            output3 = model(imgL,imgR)

        output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

        return loss

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    model.train()

    start_full_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L, batch_idx % 50==0)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))

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

