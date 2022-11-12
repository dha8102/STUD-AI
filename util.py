import numpy as np
import torch
import shutil
import os
import math
import torch.nn as nn



def scoring_softmax(pred, label, mode):
    # if mode == 'val':
    #     for q in range(len(pred)):
    #         print("pred:",pred[q])
    #         print("label:",label[q])
    #         print("===========================================")
    pred_flat = np.argmax(pred, axis=1).flatten()
    # pred_flat = pred.flatten()
    labels_flat = label.flatten()
    bunja = 0
    pre_bunmo = 0
    rec_bunmo = 0
    for i in range(len(pred_flat)):
        if pred_flat[i] == 1:
            pre_bunmo += 1
            if labels_flat[i] == 1:
                bunja += 1
        if labels_flat[i] == 1:
            rec_bunmo += 1

    acc = np.sum(pred_flat == labels_flat)
    try:
        precision = bunja / pre_bunmo
    except:
        precision = 0
    try:
        recall = bunja / rec_bunmo
    except:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0
    return acc, precision, recall, f1, bunja, pre_bunmo, rec_bunmo


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def scoring_sigmoid(pred, label):
    pred_flat = pred.flatten()
    labels_flat = label.flatten()

    for i in range(len(pred_flat)):
        if sigmoid(pred_flat[i]) >= 0.5:
            pred_flat[i] = 1
        else:
            pred_flat[i] = 0

    bunja = 0
    pre_bunmo = 0
    rec_bunmo = 0
    for i in range(len(pred_flat)):
        if pred_flat[i] == 1:
            pre_bunmo += 1
            if labels_flat[i] == 1:
                bunja += 1
        if labels_flat[i] == 1:
            rec_bunmo += 1

    acc = np.sum(pred_flat == labels_flat)
    try:
        precision = bunja / pre_bunmo
    except:
        precision = 0
    try:
        recall = bunja / rec_bunmo
    except:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0
    return acc, precision, recall, f1, bunja, pre_bunmo, rec_bunmo
