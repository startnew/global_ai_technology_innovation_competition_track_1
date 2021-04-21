#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 17:26
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : train_pre_model.py
# @Software: PyCharm
# @desc    : "根据数据集从头训练模型"
import os
from config import is_offline
from gpuServiceHelper import getAvalibleGpuList

if is_offline:
    avaliblegpus = getAvalibleGpuList()
    avaliblegpus = avaliblegpus[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(avaliblegpus)
else:
    avaliblegpus = getAvalibleGpuList()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(avaliblegpus[0])
from tokenizers.processors import BertProcessing
from tokenizers import Tokenizer
from train_tokenizers import train_tok


def get_tokenizer(tokenizer_path="../user_data/process_data/tokenizer.json"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


from torch.utils.data import Dataset
import torch
import random


class MyDataset(Dataset):
    def __init__(self, tokenizer, evaluate=False):
        self.examples = []
        src_files = ["../user_data/process_data/data/text.txt"]

        for src_file in src_files:
            print("?", src_file)
            with open(src_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                self.examples += [x.ids for x in tokenizer.encode_batch(lines)]
        random.seed(1)
        random.shuffle(self.examples)
        nums = int(len(self.examples) * 0.1)
        print("num eval", nums)
        print("num train", len(self.examples) - nums)
        if evaluate:
            self.examples = self.examples[:nums]
        else:
            self.examples = self.examples[nums:]
        self.examples = self.examples
        print("copy 10 times")
        examples_ = []
        for i in range(10):
            for x in self.examples:
                examples_.append(x)
        self.examples = examples_
        print(self.examples[0])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


###############train
from transformers import AlbertConfig

if __name__ == "__main__":
    name = '../user_data/tianchiAlbert_v2'
    tokenizer = train_tok()
    train_data_path = ""
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=120)

    train_data_set = MyDataset(tokenizer, evaluate=False)
    test_data_set = MyDataset(tokenizer, evaluate=True)
    config = AlbertConfig(
        vocab_size=1359,
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
    )
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("../user_data/tokenizer", max_len=120)
    from transformers import AlbertForMaskedLM

    if os.path.exists(name):
        try:
            model = AlbertForMaskedLM.from_pretrained(name, config=config)
        except:
            model = AlbertForMaskedLM(config=config)

    else:
        model = AlbertForMaskedLM(config=config)
    print("model_num_parameters resize before", model.num_parameters())
    model.resize_token_embeddings(len(tokenizer))
    print("model_num_parameters", model.num_parameters())
    from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

    from transformers import EarlyStoppingCallback

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    # )
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    from transformers import Trainer, TrainingArguments

    if is_offline:
        epochs = 100
        per_device_train_batch_size = 120
    else:
        epochs = 100
        per_device_train_batch_size = 180
    training_args = TrainingArguments(
        output_dir="./{}".format(name),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=2000,
        save_total_limit=2,
        learning_rate=1E-4,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data_set,
        tokenizer=tokenizer,
        eval_dataset=test_data_set,
    )
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    print(os.path.isdir(
                training_args.output_dir) ,training_args.output_dir)
    if os.path.isdir(
                training_args.output_dir) :
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("resume from :{}".format(last_checkpoint))
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model("./{}".format(name))

# python train_pre_model.py --model_name_or_path roberta-base --dataset_name MyDataset --do_train --do_eval --output_dir ./mlm_roberta/ --tokenizer_name ./models/
