import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import torch
import numpy as np
import random
import numpy as np
import torch
import torch.nn.functional as F



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

