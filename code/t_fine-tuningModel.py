#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 9:26
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : t_fine-tuningModel.py
# @Software: PyCharm
# @desc    : ""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/30 17:16
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : fine-tuningModel.py
# @Software: PyCharm
# @desc    : "基于训练好的模型进行微调"

import os
import logging

logging.basicConfig(level=logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pandas as pd
import numpy as np

from transformers import BertTokenizerFast
import torch
import math
import transformers
import time
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tools import seed_everything
from pytorch_pretrained_bert import BertAdam as Adam


from config import is_rematch,model_name,use_swa,do_swa,just_test,use_PGD,use_ema,use_lookahead,use_sda,name,model_name,use_k_fload,name,is_offline

class rank_one_DataSet(Dataset):

    def __init__(self, name="train"):
        self.name = name
        if name == "train":
            self.data_pd = tran_pd
        elif name == "test":
            self.data_pd = test_pd
        elif name == "val":
            self.data_pd = val_pd
        elif name == "sub":
            self.data_pd = submit_pd
        elif name == "all":
            self.data_pd = info_pd
        self.tokenizer = tokenizer
        self.max_len = 120

    def __getitem__(self, index):
        comment_text = self.data_pd.iloc[index]["description"]

        comment_text = " ".join([str(x) for x in comment_text])
        targets = self.data_pd.iloc[index]["labels"]

        targets_np = np.zeros(len(label_cnt))
        for label in targets:
            targets_np[label] = 1
        # print(comment_text)

        # embedding = self.tokenizer(comment_text, truncation=True, padding=True)
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        # print(inputs)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        item = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets_np, dtype=torch.float)
        }
        # print(item)
        # item["labels"] = self.data_pd.iloc[index]["labels"]
        return item

    def __len__(self):
        return self.data_pd.shape[0]

    def collate(self, samples):
        '''
        '''
        #  过滤为None的数据
        samples = list(filter(lambda x: x is not None, samples))
        # import pdb
        # pdb.set_trace()

        ids = torch.cat([x["ids"].unsqueeze(0) for x in samples], dim=0)
        # print(ids.shape)
        mask = torch.cat([x["mask"].unsqueeze(0) for x in samples], dim=0)
        token_type_ids = torch.cat([x["token_type_ids"].unsqueeze(0) for x in samples], dim=0)
        targets = torch.cat([x["targets"].unsqueeze(0) for x in samples], dim=0)
        # print(r)
        return [ids, mask, token_type_ids], targets


import torch


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        from transformers import AlbertForMaskedLM
        config = transformers.AlbertConfig.from_json_file('../user_data/tianchiAlbert_v2/config.json')
        config.output_hidden_states = True
        config.output_attentions = True

        config.num_labels = len(label_cnt)
        self.l1 = transformers.AlbertForSequenceClassification.from_pretrained('../user_data/tianchiAlbert_v2',
                                                                               config=config)

        print("self.l1", self.l1)

    def forward(self, inputs):
        ids, mask, token_type_ids = inputs[0], inputs[1], inputs[2]
        # print("ids shape",ids.shape)

        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = torch.sigmoid(output_1.logits)
        return output


def evaluate(eval_model, val_dataloader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    total_mlogloss = 0.
    count_num = 0
    count_num_all = 0
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):

            ids = data[0][0].to(device, dtype=torch.long)
            mask = data[0][1].to(device, dtype=torch.long)
            token_type_ids = data[0][2].to(device, dtype=torch.long)
            targets = data[1].to(device, dtype=torch.float)

            output = eval_model([ids, mask, token_type_ids])
            output_flat = output.view(-1, len(label_cnt))
            total_loss += len(data) * criterion(output_flat, targets).item()
            mlogloss = []
            num = 0
            p_c = output_flat.cpu().numpy()
            for simple_n, target in enumerate(targets.cpu().numpy()):
                for target_m, target_single in enumerate(target):
                    if target_m == 0:
                        continue
                    real = target_single
                    pre = p_c[simple_n][target_m]
                    if real == 0.0 and pre == 0.0:
                        real = 0.0000001
                        pre = 0.0000001
                    elif real == 1.0 and pre == 1.0:
                        real = 0.9999999
                        pre = 0.9999999
                    l = real * np.log(pre) + (1 - real) * (np.log(1 - pre))
                    if np.isnan(l):
                        print(real, pre, l)
                    mlogloss.append(l)
                    # if simple_n == 0 and target_m == 0:
                    #     print("label:{} pre:{} loss:{} mlogloss:{}".format(real, pre, l, np.mean(mlogloss)))

                    num += 1

            # print("mlogloss:{} num:{}".format(mlogloss,num))

            total_mlogloss += np.mean(mlogloss)
            count_num += 1
            count_num_all += len(data)

    # print("total_logloss:{} count_num:{}".format(total_mlogloss, count_num))

    mlogloss = -total_mlogloss / count_num

    score = 1 - mlogloss
    print("score:{} score2:{}".format(score, 1 - total_loss / count_num_all))

    return total_loss / (len(val_dataloader) * eval_batch_size - 1)


def got_result(eval_model, sub_dataloader, use_float=True):
    eval_model.eval()  # Turn on the evaluation mode

    results = []
    with torch.no_grad():
        for i, data in enumerate(sub_dataloader):
            data = data
            ids = data[0][0].to(device, dtype=torch.long)
            mask = data[0][1].to(device, dtype=torch.long)
            token_type_ids = data[0][2].to(device, dtype=torch.long)
            targets = data[1].to(device, dtype=torch.float)

            output = eval_model([ids, mask, token_type_ids])

            output_flat = output.view(-1, len(label_cnt))

            for simple_n, pre in enumerate(output_flat.cpu().numpy()):
                if use_float:
                    results.append([float(x) for x in pre[1:]])
                else:
                    # results.append(" ".join([str(float(x)) for x in pre[1:]]))
                    results.append(" ".join([str(float(x)) for x in pre[1:]]))

    return np.array(results)


if __name__ == "__main__":


    tokenizer = BertTokenizerFast.from_pretrained("../user_data/tokenizer", max_len=120, unk_token="<unk>",
                                                  mask_token="<mask>",
                                                  pad_token="<pad>", sep_token="<sep>", cls_token="<cls>")
    if is_rematch:
        test_data_path = "/tcdata/testA.csv"  # "../tcdata/track1_round1_testA_20210222.csv"
    else:
        test_data_path = "../tcdata/track1_round1_testA_20210222.csv"
    info_ = []
    # use_swa = True
    # do_swa = use_swa
    # use_k_fload = False
    # just_test = True
    # use_PGD = False  # 是否使用基于PGD的对抗训练
    # use_ema = False  # 是否是用训练过程中的指数滑动
    # use_lookahead = False  # 是否使用 lookahead
    # use_sda = True  # 是否使用SDA
    #
    # name = "{}_swa_{}_kload_{}_pgd_{}_lookahead_{}_sda_{}_isrematch_{}".format(name, do_swa, use_k_fload, use_PGD,
    #                                                                            use_lookahead, use_sda, is_rematch)
    print(name)
    with open(test_data_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                infos = line.split("|,|")
                info = {}
                info["report_id"] = int(infos[0])
                info["description"] = [int(x) for x in infos[1].split(" ")[:-1]]
                info["labels"] = [0]
                info_.append(info)
            else:
                break
    if is_offline:
        print("off line only test 1000 data")
        info_=info_[:1000]
    submit_pd = pd.DataFrame(info_)
    if is_rematch:
        train_data_path = "/tcdata/train.csv"
    else:
        train_data_path = "../tcdata/track1_round1_train_20210222.csv"
    info_ = []
    offset = 17
    with open(train_data_path, "r", encoding="utf-8") as f:
        while True:

            line = f.readline()
            # print(line)
            if line:
                infos = line.split("|,|")
                info = {}
                info["report_id"] = int(infos[0])
                info["description"] = [int(x) for x in infos[1].split(" ")[:-1]]
                if is_rematch:
                    labels = infos[2].replace("\n", "").split(",")
                    if len(labels) == 2:
                        labels_one = labels[0]
                        labels_two = labels[1]
                        labels_one = [x for x in labels_one.split(" ") if len(x) > 0]
                        labels_two = [x for x in labels_two.split(" ") if len(x) > 0]
                        labels_two = [str(int(x) + offset) for x in labels_two]
                        labels = labels_one + labels_two
                    elif len(labels) == 1:
                        labels_one = labels[0]
                        labels_one = [x for x in labels_one.split(" ") if len(x) > 0]
                        labels = labels_one
                    elif len(labels) == 0:
                        labels = [0]

                else:
                    labels = infos[2].replace("\n", "").split(" ")
                    labels = [x for x in labels if len(x) > 0]
                if len(labels) == 0:
                    labels = [0]
                else:
                    labels = [int(x) + 1 for x in labels]
                info["labels"] = labels
                info_.append(info)
            else:
                break
    # data_round_tran = pd.read_csv(train_data_path,sep="|,|")
    info_pd = pd.DataFrame(info_)
    num_empty = len([x for x in info_pd.labels if len(x) == 1 and x[0] == 0])
    num_empty_ratio = num_empty / len(info_pd.labels)
    print("empty label num:{} ratio:{}".format(num_empty, num_empty_ratio))
    label_cnt = Counter()
    FAILURE_MODE = False
    max_len_label = 0
    for label in info_pd.labels:
        l = len(label)
        max_len_label = l if l > max_len_label else max_len_label
        label_cnt.update(label)
    print("len(label_cnt)", len(label_cnt))
    desc_cnt = Counter()
    max_len = 0
    for label in info_pd.description:
        l = len(label)
        max_len = l if l > max_len else max_len
        desc_cnt.update(label)
    # tokenizer = get_tokenizer('basic_english')

    import random

    # random.seed(1)
    seed_everything(1)
    ids = list(range(info_pd.shape[0]))
    random.shuffle(ids)
    if use_k_fload:
        nflod = 5
    else:
        nflod = 1
    batch_size = 128  # 128  # 128
    eval_batch_size = 64
    sub_dataset = rank_one_DataSet(name="sub")
    sub_dataloader = DataLoader(sub_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                drop_last=False,
                                collate_fn=sub_dataset.collate)
    if is_rematch:
        tran_pd = info_pd[info_pd.report_id.isin(ids[:16000])]
        test_pd = info_pd[info_pd.report_id.isin(ids[16000:18000])]
        val_pd = info_pd[info_pd.report_id.isin(ids[18000:])]
    else:
        tran_pd = info_pd[info_pd.report_id.isin(ids[:8000])]
        test_pd = info_pd[info_pd.report_id.isin(ids[8000:9000])]
        val_pd = info_pd[info_pd.report_id.isin(ids[9000:])]
    vocab = Vocab(desc_cnt)
    train_dataset = rank_one_DataSet(name="train")
    test_dataset = rank_one_DataSet(name="test")
    val_dataset = rank_one_DataSet(name="val")
    all_dataset = rank_one_DataSet(name="all")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10,
                                  drop_last=True, collate_fn=train_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=10,
                                 drop_last=True, collate_fn=train_dataset.collate)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=10,
                                drop_last=True, collate_fn=train_dataset.collate)

    all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False,
                                collate_fn=train_dataset.collate)
    device = torch.device("cuda")
    time.sleep(10)
    model = ""
    print("empty cache")
    torch.cuda.empty_cache()
    time.sleep(10)
    model = BERTClass()
    model.to(device)
    checkpoint_dir = "../user_data/checkpoints/"
    if use_k_fload:
        checkpoint_dir = "../user_data/checkpoints/{}/kfold/".format(name)
    import copy

    print("checkpoint_dir:{}".format(checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_k_fload:
        submit_pds = []
        model_dir = "{}/".format(checkpoint_dir)
        models = []
        for x in os.listdir(model_dir):
            if ".pth" in x:
                model_file = os.path.join(model_dir, x)
                print(model_file)
                load_data = torch.load(model_file, map_location=torch.device('cpu'))
                if "state_dict" in load_data.keys():
                    models.append(load_data["state_dict"])
                else:
                    models.append(load_data)

        model_num = len(models)
        print("predict num models:{}".format(model_num))
        labels = []
        test_loss_s = []
        for model_d in models:
            model = BERTClass()
            model.load_state_dict(model_d)
            model.to(device)
            # val_loss = evaluate(best_model, val_dataloader)
            # print("val_loss",val_loss)
            results = got_result(model, sub_dataloader=sub_dataloader, use_float=True)
            test_loss = evaluate(model, test_dataloader)
            test_loss_s.append(test_loss)
            labels.append(results)
        print("test_loss_mean:{}".format(np.mean(test_loss_s)))
        labels = np.array(labels)
        print("predict labels shape", labels.shape)
        results = np.sum(labels, axis=0)
        print("predict results shape", results.shape)
        results = results / model_num
        print("predict results shape_1", results.shape)
        result_t = []
        for result in results:
            result_t.append(" ".join([str(x) for x in result]))
        submit_pd["labels"] = np.array(result_t)
        time_str = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
        save_result_path = "../result.csv"
        os.makedirs(os.path.dirname(save_result_path), exist_ok=True)
        print("save Results to :{}".format(save_result_path))
        print("submit_pd", submit_pd.head())
        submit_pd.to_csv(save_result_path, columns=["report_id", "labels"], sep=",", index=False, header=False)
        with open(save_result_path, "r", encoding="utf-8") as f:
            strs = f.read()
        with open(save_result_path, "w", encoding="utf-8") as f:
            f.write(strs.replace(",", "|,|"))
        print("save Results sucess")
    else:
        model_dir = "{}/".format(checkpoint_dir)
        models = []
        for x in os.listdir(model_dir):
            if ".pth" in x:
                model_file = os.path.join(model_dir, x)
                print(model_file)
                if name in model_file:
                    print(model_file)
                if name in model_file:
                    load_data = torch.load(model_file, map_location=torch.device('cpu'))
                    if "state_dict" in load_data.keys():
                        models.append(load_data["state_dict"])
                    else:
                        models.append(load_data)
        print("val loss")
        model.load_state_dict(models[0])
        val_loss = evaluate(model, val_dataloader)
        print("test loss")
        test_loss = evaluate(model, test_dataloader)

        results = got_result(model, sub_dataloader=sub_dataloader, use_float=False)
        submit_pd["labels"] = results

        time_str = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
        save_result_path = "../result.csv"
        os.makedirs(os.path.dirname(save_result_path), exist_ok=True)
        print("save Results to :{}".format(save_result_path))
        print("submit_pd", submit_pd.head())
        submit_pd.to_csv(save_result_path, columns=["report_id", "labels"], sep=",", index=False, header=False)
        with open(save_result_path, "r", encoding="utf-8") as f:
            strs = f.read()
        with open(save_result_path, "w", encoding="utf-8") as f:
            f.write(strs.replace(",", "|,|"))
        print("save Results sucess")
