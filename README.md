# 介绍
使用了 使用基于albert 预训练 + 微调 + swa 的解决方案
网络结构为albert base _v2 稍作修改了 attention head 的层数

-------------
注：
 tcdata下训练数据为官方提供的A,B轮的训练数据与相同的文件名称
 tcdata 
    track1_round1_testA_20210222.csv
    track1_round1_testB.csv
    track1_round1_train_20210222.csv
        
# 评测代码运行方法:
 进入到code 目录 然后执行如下命令
```bash
sh run.sh
```


# 训练代码运行方法 
进入到code 目录 然后执行如下命令
```bash
sh train.sh
``` 




