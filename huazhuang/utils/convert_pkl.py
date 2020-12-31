#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/31 4:24 下午
# @File  : convert_pkl.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :  转换torch save的pkl文件到1.2版本

import torch

def convert(oldmodel="../trained_teacher_model/macbert_teacher_max75len_5000.pkl"):
    state_dict_S = torch.load(oldmodel, map_location='cpu')
    torch.save(state_dict_S, "macbert_teacher_max75len_5000.pkl",_use_new_zipfile_serialization=False)
    print("Done")
if __name__ == '__main__':
    convert()