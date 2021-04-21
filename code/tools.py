#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 18:07
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : tools.py
# @Software: PyCharm
# @desc    : ""
import random
import os
import numpy as np
import torch
# refer from https://github.com/lonePatient/BERT-SDA/blob/c75218036ab416dd608598b1c07b79409a366aa9/tools/common.py#L43
def seed_everything(seed=1):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True