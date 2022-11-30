import sys 
sys.path.append("./")

import torch
import torch.nn as nn
import numpy as np 
from impl.metrics import binaryf1, microf1, auroc

def test_binaryf1():
    target = np.array([0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 0, 0])
    f1 = binaryf1(preds, target)
    np.testing.assert_array_equal(f1, 0.6)

def test_microf1():
    target = np.array([2, 1])
    preds = np.array([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
    f1 = microf1(preds, target)
    np.testing.assert_array_equal(f1, 0.5)

def test_auroc():
    target = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 0])
    score = auroc(preds, target)
    np.testing.assert_array_equal(score, 0.75)
    