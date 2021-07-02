# 介绍
网络结构为albert base _v2  attention head 的层数 减半

-------------
注：
 tcdata下训练数据为官方提供的A,B轮的训练数据与相同的文件名称
 tcdata 
    track1_round1_testA_20210222.csv
    track1_round1_testB.csv
    track1_round1_train_20210222.csv
    
赛事地址:
---
https://tianchi.aliyun.com/competition/entrance/531852/information\

参与时间:
----
2021-03-20 至 2021-05-11

涉及领域:
----
NLP，BERT，医学描述

成绩:
-----------
参赛队伍: 4337
初赛:64 /0.899869
复赛:45 /0.923485
第一名得分:初赛:0.927112 复赛:0.944252
实验主要使用方法:
----
基于albert 预训练(预训练中使用了lm_ngram) + 微调 + swa + smart 对抗训练 + kfload 5 
代码位置:
https://github.com/startnew/global_ai_technology_innovation_competition_track_1

主要收获总结
----
1、 NLP 领域的相关BERT 技术初步涉及，并对huggingFace有了一定的了解:
最新的一些bert模型，GLUE榜单上模型有了一定的研究与应用

2、 BERT 训练相关的对抗训练的方法，通过参考并且实现的smart对抗训练的部分实现在该项目上的应用,并完成了
一篇对应[阅读笔记](https://blog.csdn.net/Magicapprentice/article/details/115512068?spm=1001.2014.3001.5502)
代码位置:./code/smart.py

3、 预训练时，通过修改部分数据处理代码增加的ngram lm的处理方式，代码位置:./code/n_gram_maskDataCollator.py

4、 要做好充足知识储备并且保持耐心与关注比赛时间点做好合理规划。

遗憾点:
----
1、 最后基于全量数据的预训练模型的实验没有处理好
2、 多个模型结果融合没有做

实验过程中各种记录结果如下：
----


| stage| date | data| methrod | test_score | sub_score| add|
| ---- | ----  | ----  |----  |----  |----  |----  |
|rematch|2020-04-19|round one pre train + rematch fine-tuine| albert_swa_True|0.95|0.867|epoch 5|
|rematch|2020-04-19|round one pre train + rematch fine-tuine| albert_swa_True_pgdTrue|0.952|0.854|epoch 6|
|rematch|2020-04-21|round two pre train + rematch fine-tuine| albert_swa_False_pgdFalse|0.954|0.858|epoch 6|
|rematch|2020-04-21|round two pre train + rematch fine-tuine| albert_swa_False_pgdTrue|0.931|0.827|epoch 6|
|rematch|2020-04-22|round two pre train + rematch fine-tuine| albert_swa_False_smartTrue|0.962|0.872|epoch 6|
|rematch|2020-04-23|round two pre train + rematch fine-tuine| albert_swa_False_smartTrue|0.962|0.863|epoch 5|
|rematch|2020-04-23|round two pre train + rematch fine-tuine| albert_swa_False_smartTrue|0.965|0.894761|epoch 7|
|rematch|2020-04-23|round two pre train + rematch fine-tuine| albert_swa_False_smartTrue|0.966|0.89806002|epoch 8|
|rematch|2020-04-24|round two pre train + rematch fine-tuine| albert_swa_False_smartTrue|0.9678|0.90056173|epoch 9|
|rematch|2020-04-24|round two pre train + rematch fine-tuine| albert_swa_False_smartTrue|0.9707|0.902079|epoch 10|
|rematch|2020-04-25|round two pre train + rematch fine-tuine| albert_base_smartTrue|0.954|0.872929|epoch 10|
|rematch|2020-04-25|round two pre train + rematch fine-tuine| deberta_base_smartTrue|0.94844|0.872929|epoch 2|
|rematch|2020-04-26|round two pre train + rematch fine-tuine| albert_base_smartTrue|0.965770|0.895|epoch 4 new vob|
|rematch|2020-04-27|round two pre train + rematch fine-tuine| albert_base_smartTrue|0.9552|0.874|epoch 10 new vob|
|rematch|2020-04-27|round two pre train + rematch fine-tuine Adamwarm| albert_base_smartTrue|0.941|0.844|epoch 10 new vob|
|rematch|2020-04-27|round two pre train + rematch fine-tuine| albert_base_smartTrue|0.954|?|epoch 4, new vob ,smart to word embbedding,bs:64|
|rematch|2020-04-27|round two pre train + rematch fine-tuine| albert_base_smartTrue|0.953|?|epoch 10, new vob ,smart to word embbedding,bs:64|
|rematch|2020-04-28|round two pre train + rematch fine-tuine| albert_smartTrue|0.970|0.905809|epoch 6, new vob ,smart to word embbedding,bs:64|
|rematch|2020-05-04|round two pre train + rematch fine-tuine| albert_smartTrue_lookahead ngram mask|:0.9932|0.91741448|epoch max 20, new vob ,smart to word embbedding, add online test train bs:32|
|rematch|2020-05-04|round two pre train + rematch fine-tuine| albert_smartTrue_lookahead ngram mask|0.9939244185920173|0.92302893|epoch max 20, new vob ,smart to word embbedding,bs:32|
|rematch|2020-05-05|round two pre train + rematch fine-tuine| albert_smartTrue_lookahead ngram mask|0.95736|0.879|epoch max 20, new vob ,smart to word embbedding,bs:32|
|rematch|2020-05-05|round two pre train + rematch fine-tuine| albert_smartTrue_lookahead ngram mask|0.9662|0.899474|epoch max 20, new vob ,smart to word embbedding,bs:32|

前几名的解决方案
----
todo

        
# 运行方法:
```bash
pip install -r ./requirements.txt
```
 
```bash
sh run.sh
```
