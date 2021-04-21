#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 14:58
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : config.py
# @Software: PyCharm
# @desc    : ""
use_swa = False
do_swa = use_swa
use_k_fload = False
just_test = False
use_PGD = False  # 是否使用基于PGD的对抗训练
use_ema = False  # 是否是用训练过程中的指数滑动
use_lookahead = False  # 是否使用 lookahead
use_sda = False  # 是否使用SDA
is_rematch = True
model_name = "fine_tuning_albert"
model = ""
add_freeze = True
add_round_one_data = False # 是否加入round one 的数据进行训练
name = "{}_swa_{}_kload_{}_pgd_{}_lookahead_{}_sda_{}_isrematch_{}_add_freeze_{}_add_round_one_data_{}".format(model_name, do_swa,
                                                                                         use_k_fload, use_PGD,
                                                                                         use_lookahead, use_sda,
                                                                                         is_rematch, add_freeze, add_round_one_data)

#is_offline = False
is_offline = True  # 本地
