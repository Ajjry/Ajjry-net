# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np

from color_transfer import ColorTransfer

import prepara

def demo(img_names_str,ref_names_str):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    img_folder = os.path.join(cur_dir)
    img_names = [
        img_names_str
    ]
    ref_names = [
        ref_names_str
    ]
    '''
    out_names = [
        out_names_str
    ]
    '''
    img_paths = [os.path.join(img_folder, x) for x in img_names]
    ref_paths = [os.path.join(img_folder, x) for x in ref_names]
    #out_paths = [os.path.join(img_folder, x) for x in out_names]
    # cls init
    PT = ColorTransfer(n=300)

    for img_path, ref_path in zip(
            img_paths, ref_paths):
        img_arr_in = cv2.imread(img_path)
        img_arr_ref = cv2.imread(ref_path)
        img_arr_in=prepara.preparation(img_arr_ref,img_arr_in)     
        img_arr_lt = PT.lab_transfer(
            img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        return img_arr_lt

