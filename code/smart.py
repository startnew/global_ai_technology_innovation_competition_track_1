#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 15:32
# @Author  : zhuzhaowen
# @email   : shaowen5011@gmail.com
# @File    : smart.py
# @Software: PyCharm
# @desc    : "smart 对抗训练"

import torch
import torch.nn.functional as F
import torch.nn as nn

def generate_noise(embed, mask, epsilon=1e-5,ln =None):
	#生成与embed 同尺寸方差为epsion的符合正态分布的noise
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    if ln:
        print("ln:before",noise.shape)
        noise = ln(noise)
        print("ln:after", noise.shape)
        print("ln:after", noise)


    noise.requires_grad_()
    return noise

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    if reduce:
        return (p* (rp- ry) * 2).sum() / bs
    else:
        return (p* (rp- ry) * 2).sum()

class SmartPerturbation():
    def __init__(self,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 loss_map=[],
                 norm_level=0,emb_size=256):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta 更新扰动后的x_i的学习率
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma 生成扰动噪音的方差
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        self.ln_noise = nn.LayerNorm(emb_size, eps=1e-12)  # 参考deberta 增加 layerNorm
        assert len(loss_map) > 0



    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
    	# 计算梯度 以及 有效梯度的 方向
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction

    def forward(self, model,
                logits,
                input_ids,
                token_type_ids,
                attention_mask,
                premise_mask=None,
                hyp_mask=None,
                task_id=0,# task_id是多任务的 id 由实际任务决定 如果没有则可以不用考虑
                pairwise=1):
        # adv training
        # 这个参数根据模型forward 模块参数的顺序 做调整
        vat_args = [[input_ids,attention_mask, token_type_ids, ], premise_mask, hyp_mask, task_id, 1]
        # fwd_type 为 1 在构建模型forward时只输出模型的embed 信息

        # init delta
        # 输出 embded
        embed = model(*vat_args)
        print(self.ln_noise.normalized_shape,embed.shape)

        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var,ln=self.ln_noise)
        for step in range(0, self.K):
            # 需要注意的是
            vat_args = [[input_ids,attention_mask ,token_type_ids], premise_mask, hyp_mask, task_id, 2, embed + noise]
            # 使用加入噪音的embed 输出预测结果
            adv_logits = model(*vat_args)
            # 排序或者分类使用kl散度衡量两者之间的差异
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            #  分布损失与 扰动之间的梯度
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            # 梯度的范数
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            # 更新到主要训练过程中的梯度 为扰动与原始输出差异损失对扰动求出的梯度 乘以 扰动的学习率
            eff_delta_grad = delta_grad * self.step_size
            #
            delta_grad = noise + delta_grad * self.step_size
            noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()
        vat_args = [[input_ids,attention_mask ,token_type_ids], premise_mask, hyp_mask, task_id, 2, embed + noise]
        adv_logits = model(*vat_args)
        adv_lc = self.loss_map[task_id]
        adv_loss = adv_lc(logits, adv_logits)
        return adv_loss, embed.detach().abs().mean(), eff_noise.detach().abs().mean()

from torch.nn.modules.loss import _Loss
class Criterion(_Loss):
    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        return

class BCeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Binerary Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name
        self.func = torch.nn.BCELoss()
    def forward(self, input, target, weight=None, ignore_index=-1,reduction='batchmean'):
        """weight: sample weight
        """
        input = input.float()
        target = target.float()
        #print(input.shape,target.shape)
        #loss =  self.func(torch.sigmoid(input), torch.sigmoid(target))
        loss = F.mse_loss(torch.sigmoid(input), torch.sigmoid(target))
        loss = loss * self.alpha
        return loss

if __name__ == "__main__":
    # usege case
    config={}
    config["adv_epsilon"] = 1e-6
    config["multi_gpu_on"] = False
    config['adv_step_size'] =1e-5
    config['adv_noise_var']=1e-5
    config['adv_p_norm'] = 'inf'
    config['adv_k'] = 1
    config['fp16'] = False
    adv_task_loss_criterion = [CeCriterion()]
    config["adv_norm_level"] = 0

    adv_teacher = SmartPerturbation(config['adv_epsilon'],
                    config['multi_gpu_on'],
                    config['adv_step_size'],
                    config['adv_noise_var'],
                    config['adv_p_norm'],
                    config['adv_k'],
                    config['fp16'],
                    loss_map=adv_task_loss_criterion,
                    norm_level=config['adv_norm_level'])

    # adv_inputs = [self.mnetwork, logits] + inputs + [task_type, batch_meta.get('pairwise_size', 1)]
    # adv_loss, emb_val, eff_perturb = self.adv_teacher.forward(*adv_inputs)
    # loss = loss + self.config['adv_alpha'] * adv_loss

