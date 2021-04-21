# -*- coding: utf-8 -*-
'''
____________________________________________________________________
    File Name   :    gpuServiceHelper
    Description :    
    Author      :    zhaowen
    date        :    2019/6/24
____________________________________________________________________
    Change Activity:
                        2019/6/24:
____________________________________________________________________
         
'''
__author__ = 'zhaowen'
import os
import random
import hashlib
import time
import sys

def getAvalibleGpuList(min_memory=8096):
    """
    获取可用gpu列表只能用于linux系统中（注意）
    :param min_memory:
    :return:
    """
    print("获取可以使用的GPU--")
    if sys.platform == "linux":
        k = 0

        strrannum = str(random.randint(0, 100))

        strtimenum = str(time.time())
        rantime = "".join((strrannum, strtimenum))
        while not os.path.exists("gputmp_{}".format(rantime)):
            k += 1
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gputmp_{}'.format(rantime))
            time.sleep(0.1)
            if k > 10:
                break

        print("tmp:exists{}".format(os.path.exists("gputmp_{}".format(rantime))))
        memory_gpu = [int(x.split()[2]) for x in open('gputmp_{}'.format(rantime), 'r').readlines()]
        avaliable_devices = []
        for i, m in enumerate(memory_gpu):
            if m > min_memory:
                print("gpu:{},memory:{}".format(i, m))
                avaliable_devices.append(str(i))
        os.system('rm gputmp_{}'.format(rantime))
        print("删除gputmp_{}".format(rantime), os.path.exists("gputmp_{}".format(rantime)))
        return avaliable_devices
    else:
        print("不支持在非linux系统的环境下运行{},返回默认GPU0可用".format(sys.platform))
        avaliable_devices = ["0"]
        return avaliable_devices


if __name__ == "__main__":
    print("可用的GPU列表：", getAvalibleGpuList())
