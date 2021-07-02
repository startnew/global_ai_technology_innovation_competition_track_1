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
use_official_train = True
model = ""
add_freeze = False
use_smart = True  # 是否使用基于smart的对抗训练
use_apex = True    # 是否使用apex 进行训练加速
use_sift = True  # 是否使用Scale Invariant Fine Tuning 进行训练
add_round_one_data = False  # 是否加入round one 的数据进行训练
name = "{}_swa_{}_kload_{}_pgd_{}_lookahead_{}_sda_{}_isrematch_{}_add_freeze_{}_add_round_one_data_{}_{}".format(
    model_name, do_swa,
    use_k_fload, use_PGD,
    use_lookahead, use_sda,
    is_rematch, add_freeze, add_round_one_data, use_smart)
import transformers
from transformers import AlbertConfig,DebertaV2Config

infos = [
    {"name": "albert",
     "config": AlbertConfig(
         vocab_size=1356,
         embedding_size=256,
         hidden_size=768,
         num_hidden_layers=6,
         num_attention_heads=12,
         intermediate_size=3072,
         hidden_act="gelu_new",
         hidden_dropout_prob=0.0,
         num_hidden_groups=1,
         attention_probs_dropout_prob=0.0,
         classifier_dropout_prob=0.1,
         down_scale_factor=1,
         max_position_embeddings=512,
         type_vocab_size=2,
         initializer_range=0.02,
         layer_norm_eps=1e-12,
     ),
     "mlmModel": transformers.AlbertForMaskedLM,
     "clsModel": transformers.AlbertForSequenceClassification,
     "word_embed": "albert"
     },
    {"name": "albert_base",
     "config": AlbertConfig(
         vocab_size=1359,
         embedding_size=256,
         hidden_size=768,
         num_hidden_layers=12,
         num_attention_heads=12,
         intermediate_size=3072,
         hidden_act="gelu_new",
         hidden_dropout_prob=0.0,
         num_hidden_groups=1,
         attention_probs_dropout_prob=0.0,
         classifier_dropout_prob=0.1,
         down_scale_factor=1,
         max_position_embeddings=512,
         type_vocab_size=2,
         initializer_range=0.02,
         layer_norm_eps=1e-12,
     ),
     "mlmModel": transformers.AlbertForMaskedLM,
     "clsModel": transformers.AlbertForSequenceClassification,
     "word_embed": "albert"
     },
    {"name": "debertav2_base",
     "config": DebertaV2Config(
            vocab_size=1359,
            embedding_size=256,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.0,
            num_hidden_groups=1,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            down_scale_factor=1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-7,
            position_biased_input=False,
            max_relative_positions=-1,
            relative_attention=True,
            pos_att_type="c2p|p2c",
        ),
     "mlmModel": transformers.DebertaV2ForMaskedLM,
     "clsModel": transformers.DebertaV2ForSequenceClassification,
     "word_embed": "deberta"
     }


]

epochs = 6

is_offline = False
# is_offline = True  # 本地
