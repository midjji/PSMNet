import torch
import torch.nn.parallel
import torch.utils.data
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


from .models import stackhourglass, basic, feature_extraction




def get_pretrained_stackhourglass( settings):


    path=os.getcwd() + settings['pretrained_models_path']

    if 'force_override_model' in settings:
        model = settings['force_override_model'](settings)
    else:
        model = stackhourglass(settings['n_depths'])

    model = nn.DataParallel(model)
    model.cuda()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'])
    model=model.module.cpu()
    return model


def get_pretrained_feature_extractor(settings):

    model=get_pretrained_stackhourglass(settings)

    return model.feature_extraction

def get_psmnet_feature_extractor(settings):
    if settings['load_pretrained_feature_extractor']:
        return get_pretrained_feature_extractor(settings)
    return feature_extraction()

