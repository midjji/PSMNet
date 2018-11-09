import torch
import torch.nn.parallel
import torch.utils.data
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


from .models import stackhourglass, basic


class PSMNET_WEIGHTS:
    def __init__(self):
        self.stackedhourglass=None

    def get_pretrained_stackhourglass(self, name="others_code/PSMNet/pretrained/pretrained_sceneflow.tar"):
        if self.stackedhourglass is None:
            cwd = os.getcwd()

            self.stackedhourglass=stackhourglass(192)
            self.stackedhourglass = nn.DataParallel(self.stackedhourglass)
            self.stackedhourglass.cuda()
            state_dict = torch.load(cwd+"/"+name)
            self.stackedhourglass.load_state_dict(state_dict['state_dict'])
        return self.stackedhourglass
    def get_feature_extractor(self):

        model=self.get_pretrained_stackhourglass()
        return model.feature_extraction








