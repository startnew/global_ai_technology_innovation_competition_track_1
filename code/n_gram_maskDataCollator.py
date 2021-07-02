#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 16:11
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : n_gram_maskDataCollator.py
# @Software: PyCharm
# @desc    : "n-gram 的 data mask"
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
from transformers import BatchEncoding, PreTrainedTokenizerBase
import random
from transformers.data.data_collator import _collate_batch,tolist
@dataclass
class DataCollatorForWholeWordMask_Ngram(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. with n gram n is 3
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _collate_batch(input_ids, self.tokenizer)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _collate_batch(mask_labels, self.tokenizer)
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        # if random.random()<0.001:
        #     print({"input_ids": inputs, "labels": labels})
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # 输入文字token的在当前句子中的位置列表 去除一些特殊标记后的 [[0],[1],...]
        cand_indexes = []
        all_infos = set()
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            all_infos.add(i)

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)

            else:
                cand_indexes.append([i])


        # 将顺序打乱 [[3],[1],...]
        random.shuffle(cand_indexes)
        # 预测的个数为: 当前句子的长度×mask的概率
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        max_ngram = 3
        select_ngram_num = random.choice([x+1 for x in list(range(max_ngram))])
        num_to_predict = num_to_predict * select_ngram_num
        covered_indexes = set()
        for index_set in cand_indexes:
            # 大于需要预测的个数了直接跳出
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            #
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            # 被mask 过的字/词 后面直接跳过不继续mask
            if is_any_index_covered:
                continue
            for index in index_set:
                for offset_n in range(select_ngram_num):
                    # ngram mask
                    if index+offset_n not in covered_indexes and index+offset_n in all_infos:
                        covered_indexes.add(index+offset_n)
                        masked_lms.append(index+offset_n)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
