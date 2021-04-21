#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 16:57
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : train_tokenizers.py
# @Software: PyCharm
# @desc    : "使用tokenizer训练toknenizer"
from pathlib import Path
#from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import os

def train_tok(use_sentencepiece=False):
    data_dir = "/tcdata/"
    files = [os.path.join(data_dir,x) for x in os.listdir(data_dir) if ".csv" in x]
    info_ = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    infos = line.split("|,|")
                    info = {}
                    info["report_id"] = int(infos[0])
                    info["description"] = [int(x) for x in infos[1].split(" ")[:-1]]
                    info["description"] = " ".join([str(x) for x in info["description"]])
                    info["labels"] = [0]
                    info_.append(info)
                else:
                    break

    data_pd = pd.DataFrame(info_)
    info_pd=data_pd

    p = "../user_data/process_data/data/text.txt"
    os.makedirs(os.path.dirname(p),exist_ok=True)

    with open("../user_data/process_data/data/text.txt", "w", encoding="utf-8") as f:
        print("num data:{}".format(len(data_pd.description.unique())))
        for label in data_pd.description.unique():
            f.write(label + "\n")
    if use_sentencepiece:
        import sentencepiece as spm
        os.makedirs("../user_data/spm_tokenizer", exist_ok=True)
        spm.SentencePieceTrainer.train(input='../user_data/process_data/data/text.txt', model_prefix='m', vocab_size=177)


    else:
        from tokenizers import BertWordPieceTokenizer
        tokenizer = BertWordPieceTokenizer()
        special_tokens = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]
        tokenizer.train([p], min_frequency=2, special_tokens=special_tokens)
        tokenizer.add_special_tokens(special_tokens)
        sample = info_pd.description[0]
        strs = " ".join([str(x) for x in sample])
        print(strs)
        encoded = tokenizer.encode(strs)
        print("print(encoded.ids)", encoded.ids)
        print("print(encoded.tokens)", encoded.tokens)
        os.makedirs("../user_data/tokenizer",exist_ok=True)
        tokenizer.save_model("../user_data/tokenizer")
        print("get_vocab_size", tokenizer.get_vocab_size())
        return tokenizer
if __name__ == "__main__":
    train_tok()






