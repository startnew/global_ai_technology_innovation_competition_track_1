#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/28 15:04
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : official_fine_tune.py
# @Software: PyCharm
# @desc    : ""

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from fine_tuningModel import BERTClass, BertTokenizerFast, is_rematch, pd, name, use_smart, SmartPerturbation, Counter, \
    seed_everything, use_k_fload, rank_one_DataSet, DataLoader, just_test, Vocab, time, torch


def model_init():
    return BERTClass()


criterion = torch.nn.BCELoss()
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    score = 1 - criterion(predictions, labels)
    return score


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
    batch_size = 64  # 128  # 128
    eval_batch_size = 64
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
        metric_name = "accuracy"
        args = TrainingArguments(
            "test-glue",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
        )
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator = train_dataset.collate(),
        )
