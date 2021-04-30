#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: October 11, 2019
# Email: rkim@salk.edu
# Description: Contains several general-purpose utility functions

import os
import tensorflow as tf
import argparse

def set_gpu(gpu, frac):
    """
    Function to specify which GPU to use

    INPUT
        gpu: string label for gpu (i.e. '0')
        frac: gpu memory fraction (i.e. 0.3 for 30% of the total memory)
    OUTPUT
        tf sess config
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    return gpu_options

def restricted_float(x):
    """
    Helper function for restricting input arg to range from 0 to 1

    INPUT
        x: string representing a float number

    OUTPUT
        x or raises an argument type error
    """
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r no in range [0.0, 1.0]"%(x,))
    return x

def str2bool(v):
    """
    Helper function to parse boolean input args

    INPUT
        v: string representing true or false
    OUTPUT
        True or False or raises an argument type error
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




