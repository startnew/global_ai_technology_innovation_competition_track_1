#!/usr/bin/env python
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
from gpuServiceHelper import getAvalibleGpuList

avaliblegpus = getAvalibleGpuList()
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(avaliblegpus[0])
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
from torchcontrib.optim import SWA

from optimizer import Lookahead
from tools import seed_everything
from pytorch_pretrained_bert import BertAdam as Adam
from config import is_rematch, do_swa, just_test, use_PGD, use_ema, use_lookahead, use_sda, name, \
    use_k_fload, add_freeze, add_round_one_data, use_smart,epochs,use_sift
from smart import SmartPerturbation
from ema import EMA


class rank_one_DataSet(Dataset):

    def __init__(self, name="train",data_pd=None,tokenizer_n=None):
 
        self.name = name
        if data_pd is not None:
            self.data_pd = data_pd
       
        elif name == "train":
            self.data_pd = tran_pd
        elif name == "test":
            self.data_pd = test_pd
        elif name == "val":
            self.data_pd = val_pd
        elif name == "sub":
            self.data_pd = submit_pd
        elif name == "all":
            self.data_pd = info_pd
        if tokenizer_n is not None:
            self.tokenizer = tokenizer_n
        else:
            self.tokenizer = tokenizer
        self.max_len = 120

    def __getitem__(self, index):
        comment_text = self.data_pd.iloc[index]["description"]

        comment_text = " ".join([str(x) for x in comment_text])
        targets = self.data_pd.iloc[index]["labels"]

        targets_np = np.zeros(30)
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
    def __getitembak__(self, index):
        comment_text = self.data_pd.iloc[index]["description"]

        comment_text = " ".join([str(x) for x in comment_text])
        targets = self.data_pd.iloc[index]["labels"]

        targets_np = np.zeros(30)
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
        ids= torch.tensor(ids, dtype=torch.long),
        mask= torch.tensor(mask, dtype=torch.long),
        token_type_ids=torch.tensor(token_type_ids, dtype=torch.long)
        targets= torch.tensor(targets_np, dtype=torch.float)
        inputs = [ids,mask,token_type_ids]
        item = {

            'targets': torch.tensor(targets_np, dtype=torch.float),
            "inputs":[ids,mask,token_type_ids]
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
         
	
        device = torch.device("cpu")
        ids = torch.cat([x["ids"].unsqueeze(0) for x in samples], dim=0)
        # print(ids.shape)
        mask = torch.cat([x["mask"].unsqueeze(0) for x in samples], dim=0)
        token_type_ids = torch.cat([x["token_type_ids"].unsqueeze(0) for x in samples], dim=0)
        targets = torch.cat([x["targets"].unsqueeze(0) for x in samples], dim=0)
        # print(r)
        ids = ids.to(device,dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        inputs = [ids, mask, token_type_ids]
        return {"inputs":inputs, "labels":targets}


import torch


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        from transformers import AlbertForMaskedLM
        pre_model_path = "../user_data/tianchiAlbert_v2"
        #pre_model_path = "../user_data/tianchiAlbert_v2_base"
        config = transformers.AlbertConfig.from_json_file(os.path.join(pre_model_path,'config.json'))
        config.output_hidden_states = True
        config.output_attentions = True

        config.num_labels =30# len(label_cnt)
        self.l1 = transformers.AlbertForSequenceClassification.from_pretrained(pre_model_path,
                                                                               config=config)
        print(self.l1)
        if add_freeze:
            print("add freeze")
            for name, p in self.l1.named_parameters():
                if "classifier" in name:
                    p.requires_grad = True
                elif "pooler" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
#        embedding_output = self.l1.albert.embeddings(input_ids, token_type_ids)
        embedding_output = self.l1.albert.embeddings.word_embeddings(input_ids)
        return embedding_output



    def forward(self, inputs, premise_mask=None, hyp_mask=None, task_id=0, fwd_type=0, embed=None,labels=None,**kwargs):
        # premise_mask=None, hyp_mask=None, task_id=0 在这个任务中 这几个参数没有意义，只是和smart上参数对应懒得改
        ids, mask, token_type_ids = inputs[0], inputs[1], inputs[2]
        device = torch.device("cuda")       
        ids = ids.to(device,dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)
    #    labels = labels.to(device,dtype=torch.float)
        if fwd_type == 2:
            # fwd_type 为2 时只返回使用干扰后的embedding 信息 的预测结果
            assert embed is not None
            output_1 = self.l1(input_ids=None, token_type_ids=token_type_ids, attention_mask=mask, inputs_embeds=embed)
            return output_1.logits
        elif fwd_type == 1:
            # fwd_type 为1 时只返回embedding信息
            return self.embed_encode(ids, token_type_ids, mask)
        else:
            output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = torch.sigmoid(output_1.logits)
        output = output.view(-1, 30)
        criterion = torch.nn.BCELoss()
        if labels is not None:
            return {"loss":criterion(output, labels),"logits":output}
        return output

def train(train_dataloader, global_step=0):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, dat in enumerate(train_dataloader):

        optimizer.zero_grad()

        data = dat
        ids = data[0][0].to(device, dtype=torch.long)
        mask = data[0][1].to(device, dtype=torch.long)
        token_type_ids = data[0][2].to(device, dtype=torch.long)
        targets = data[1].to(device, dtype=torch.float)
        inputs = [ids, mask, token_type_ids]
        output = model(inputs)
        logits = output.view(-1, len(label_cnt))
        loss = criterion(logits, targets)
        if use_smart:
            adv_inputs = [model, logits, ] + [ids, token_type_ids, mask]

            adv_loss, emb_val, eff_perturb = adv_teacher.forward(*adv_inputs)
            loss = loss + config['adv_alpha'] * adv_loss

        if use_sda:  # 使用自集成
            with torch.no_grad():
                kd_logits = kd_model([ids, mask, token_type_ids])
                kd_loss = kd_loss_fct(output, kd_logits)
                kd_coeff = 1.0
                loss += kd_coeff * kd_loss

        loss.backward()
        if use_PGD:
            pgd.backup_grad()  # 保存正常的grad
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    optimizer.zero_grad()
                else:
                    pgd.restore_grad()  # 恢复正常的grad
                output_1 = model([ids, mask, token_type_ids])
                loss_sum = criterion(output_1.view(-1, len(label_cnt)), targets)
                loss_sum.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 在embedding上添加对抗扰动, first attack时备份param.data

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1
        if use_sda:
            decay = 0.995

            decay = min(decay, (1 + global_step) / (10 + global_step))

            one_minus_decay = 1.0 - decay
            with torch.no_grad():
                parameters = [p for p in model.parameters() if p.requires_grad]
                for s_param, param in zip(kd_model.parameters(), parameters):
                    s_param.sub_(one_minus_decay * (s_param - param))

        total_loss += loss.item()
        log_interval = 20
        # print(loss.item())
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_dataloader), scheduler.get_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return global_step


def evaluate(eval_model, val_dataloader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    total_mlogloss = 0.
    count_num = 0
    count_num_all = 0
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
        pass
    else:
        test_data_path = "../tcdata/track1_round1_testA_20210222.csv"
        info_ = []
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

            submit_pd = pd.DataFrame(info_)
    info_ = []

    print(name)
    if is_rematch:
        train_data_path = "../tcdata/train.csv"
    else:
        train_data_path = "../tcdata/track1_round1_train_20210222.csv"

    info_ = []
    offset = 17
    if use_smart:
        print("use_smart train")
        config = {}
        config["adv_epsilon"] = 1e-6
        config["multi_gpu_on"] = False
        config['adv_step_size'] = 1e-5
        config['adv_noise_var'] = 1e-5
        config['adv_p_norm'] = 'inf'
        config['adv_k'] = 1
        config['fp16'] = False
        from smart import BCeCriterion
        adv_task_loss_criterion = [BCeCriterion()]
        config["adv_norm_level"] = 0
        config["adv_alpha"] = 1

        adv_teacher = SmartPerturbation(config['adv_epsilon'],
                                        config['multi_gpu_on'],
                                        config['adv_step_size'],
                                        config['adv_noise_var'],
                                        config['adv_p_norm'],
                                        config['adv_k'],
                                        config['fp16'],
                                        loss_map=adv_task_loss_criterion,
                                        norm_level=config['adv_norm_level'])
    with open(train_data_path, "r", encoding="utf-8") as f:
        while True:

            line = f.readline()
            # print(line)
            if line:
                infos = line.split("|,|")
                info = {}
                info["report_id"] = int(infos[0])
                info["description"] = [int(x) for x in infos[1].split(" ")[:-1]]
                info["label_ori"] = infos[2]

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
                        print("find only one type info:{}".format(infos[2]))
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
    print(info_pd[info_pd["report_id"] == 13426])
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
    import random

    seed_everything(1)
    ids = list(range(info_pd.shape[0]))
    random.shuffle(ids)
    if use_k_fload:
        nflod = 5
    else:
        nflod = 1
    batch_size = 32#64  # 128  # 128
    eval_batch_size = 64
    if is_rematch:
        pass
    else:
        sub_dataset = rank_one_DataSet(name="sub")
        sub_dataloader = DataLoader(sub_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                    drop_last=False,
                                    collate_fn=sub_dataset.collate)
    for nflod_i in range(nflod):
        global_step = 0
        if just_test and nflod_i > 1:
            continue
        if nflod == 1:
            if is_rematch:
                split_ratio = 0.1
                num_train = int(len(ids) * (1 - split_ratio))
                num_val = int(int(len(ids) - num_train) / 2)
                print("num_train:{},num_test:{}".format(num_train, num_val))
                tran_pd = info_pd[info_pd.report_id.isin(ids[:num_train])]
                test_pd = info_pd[info_pd.report_id.isin(ids[num_train:num_train + num_val])]
                val_pd = info_pd[info_pd.report_id.isin(ids[num_train + num_val:])]
            else:
                tran_pd = info_pd[info_pd.report_id.isin(ids[:8000])]
                test_pd = info_pd[info_pd.report_id.isin(ids[8000:9000])]
                val_pd = info_pd[info_pd.report_id.isin(ids[9000:])]
        else:
            print("Kfload No.:{}".format(nflod_i))
            one_flod_num = int(info_pd.shape[0] / nflod)
            print("one_flod_num", one_flod_num)
            test_ids = ids[nflod_i * one_flod_num:(nflod_i + 1) * one_flod_num]

            val_ids = test_ids[:int(one_flod_num / 2)]
            test_ids = test_ids[:int(one_flod_num / 2)]
            train_ids = [x for x in ids if x not in test_ids]
            # if is_offline:
            #     train_ids=train_ids[:1200]
            print("train_num", len(train_ids))
            print("val_num", len(val_ids))
            print("test_num", len(test_ids))
            tran_pd = info_pd[info_pd.report_id.isin(train_ids)]
            test_pd = info_pd[info_pd.report_id.isin(test_ids)]
            val_pd = info_pd[info_pd.report_id.isin(val_ids)]
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
        if use_ema:
            ema = EMA(model, decay=0.999)
        if use_PGD:
            from pgd import PGD

            pgd = PGD(model)
            K = 3
        criterion = torch.nn.BCELoss()
        lr = 5e-5
        print("lr:{}".format(lr))
        #epochs = 6  # The number of epochs
        early_stop = max(1, epochs // 10)
        base_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))
        steps_per_epoch = len(train_dataset) / batch_size
        #base_opt = Adam(model.parameters(), lr=lr, warmup=0.1, b1=0.99, b2=0.999, t_total=epochs * steps_per_epoch,)
        if do_swa:
            steps_per_epoch = len(train_dataset) / batch_size
            steps_per_epoch = int(steps_per_epoch) - 1
            print("do_swa:{}".format(steps_per_epoch))
            swa_start_epoch = min(epochs - 1, np.round(epochs * 0.8))
            print("swa from ", swa_start_epoch)
            optimizer = SWA(base_opt, swa_start=swa_start_epoch * steps_per_epoch, swa_freq=steps_per_epoch,
                            swa_lr=lr / 2)
        else:
            optimizer = base_opt
        if use_lookahead:
            optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

        # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        # lr_finder.range_test(train_dataloader, val_loader=val_dataloader, end_lr=1, num_iter=100, step_mode="linear")
        # #fig = lr_finder.plot().get_figure()
        # #fig.savefig("learn_{}.png".format(name))
        # lr_finder.reset()

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.999)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                          last_epoch=-1)

        best_val_loss = float("inf")

        print("early_stop:{}".format(early_stop))
        not_update = 0
        # best_model = None
        checkpoint_dir = "../user_data/checkpoints/"
        if use_k_fload:
            checkpoint_dir = "../user_data/checkpoints/{}/kfold/".format(name)
        import copy

        os.makedirs(checkpoint_dir, exist_ok=True)
        if use_k_fload:
            save_path = "{}/{}_best_model_{}.pth".format(checkpoint_dir, name, nflod_i)
        else:
            save_path = "{}/{}_best_model.pth".format(checkpoint_dir, name)
        epoch_start = 1
        if os.path.exists(save_path):
            try:
                load_data = torch.load(save_path, map_location=torch.device('cpu'))
                if "state_dict" in load_data.keys():
                    ckpt = load_data
                    model.load_state_dict(load_data["state_dict"])
                    best_model = copy.deepcopy(model)
                    state_dict = ckpt['state_dict']
                    optimizer_dict = ckpt['optimizer']
                    epoch_start = ckpt['epoch']
                    if "best_val_loss" in ckpt.keys():
                        best_val_loss = ckpt["best_val_loss"]
                    optimizer.load_state_dict(optimizer_dict)
                    scheduler.step(epoch_start)
                    print("load model from :{}".format(save_path))
                else:
                    model.load_state_dict(load_data)
                    best_model = copy.deepcopy(model)

            except Exception as e:
                print("e:{}".format(e))
                pass
        if just_test:
            best_swa = "{}/{}_best_swa.pth".format(checkpoint_dir, name)
            if os.path.exists(best_swa):
                if not do_swa:
                    PATH = "{}/{}_best.pth".format(checkpoint_dir, name)
                else:
                    PATH = "{}/{}_best_swa.pth".format(checkpoint_dir, name)
                print("load from:{}".format(PATH))
                load_data = torch.load(PATH, map_location=torch.device('cpu'))
                if "state_dict" in load_data.keys():
                    model.load_state_dict(load_data["state_dict"])
                else:
                    model.load_state_dict(load_data)
                model.eval()
                best_model = model
            else:
                if do_swa:
                    model_dir = "{}/swa/".format(checkpoint_dir)
                    models = []
                    for x in os.listdir(model_dir):
                        if ".pth" in x:
                            load_data = torch.load(os.path.join(model_dir, x), map_location=torch.device('cpu'))
                            if "state_dict" in load_data.keys():
                                models.append(load_data["state_dict"])
                            else:
                                models.append(load_data)

                    model_num = len(models)
                    print("SWA models num:{}".format(model_num))
                    model_keys = models[-1].keys()  # ['state_dict']
                    state_dict = models[-1]  # ['state_dict']
                    new_state_dict = state_dict.copy()
                    ref_model = models[-1]
                    for key in model_keys:
                        sum_weight = 0.0
                        for m in models:
                            sum_weight += m[key]  # ['state_dict']
                        avg_weight = sum_weight / model_num
                        new_state_dict[key] = avg_weight
                    ref_model = new_state_dict  # ['state_dict']
                    torch.save(ref_model, "{}/{}_best_swa.pth".format(checkpoint_dir, name))
                    print("swa model path:{}".format("{}/{}_best_swa.pth".format(checkpoint_dir, name)))
                    best_model = model.load_state_dict(ref_model)
        else:
            if epoch_start == epochs:
                pass
            else:
                if use_sda:
                    from torch.nn import MSELoss

                    kd_loss_fct = MSELoss()
                    kd_model = copy.deepcopy(model)
                    kd_model.eval()

                for epoch in range(epoch_start, epochs + 1):
                    epoch_start_time = time.time()
                    train(train_dataloader)
                    # import pdb
                    # pdb.set_trace()
                    # print(model.l1.classifier.weight[0, :3].detach())
                    print("swa before:")
                    if use_lookahead:
                        optimizer._backup_and_load_cache()
                    val_loss = evaluate(model, val_dataloader)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                          'valid ppl {:8.2f} not update:{:3d}'.format(epoch, (time.time() - epoch_start_time),
                                                                      val_loss, math.exp(val_loss), not_update))
                    if do_swa and epoch > swa_start_epoch:
                        print("start swa epoch:", swa_start_epoch)
                        print("swa before", model.l1.classifier.weight[0, :3].detach())
                        optimizer.swap_swa_sgd()
                        print("swa after:")
                        print("swa first", model.l1.classifier.weight[0, :3].detach())
                        print("bn_update")
                        optimizer.bn_update(train_dataloader, model, device='cuda')
                        print("swa second bn update after", model.l1.classifier.weight[0, :3].detach())

                        val_loss = evaluate(model, val_dataloader)

                        print('-' * 89)
                        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                              'valid ppl {:8.2f} not update:{:3d}'.format(epoch, (time.time() - epoch_start_time),
                                                                          val_loss, math.exp(val_loss), not_update))
                        optimizer.swap_swa_sgd()
                    if use_lookahead:
                        optimizer._clear_and_load_backup()

                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                          'valid ppl {:8.2f} not update:{:3d}'.format(epoch, (time.time() - epoch_start_time),
                                                                      val_loss, math.exp(val_loss), not_update))
                    print('-' * 89)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 "best_val_loss": best_val_loss}
                        torch.save(state, save_path)
                        print("save_best")
                        not_update = 0
                    else:
                        not_update += 1
                        if not_update > early_stop:
                            print("not_update:{} earyly_stop{},best_val_loss:{}".format(not_update, early_stop,
                                                                                        best_val_loss))
                            model = copy.deepcopy(best_model)
                            best_model = model
                            break
                if do_swa:
                    optimizer.swap_swa_sgd()
