import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import torchsummaryX
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import glob
import wandb
import yaml
import json
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup

# Check that MPS is available
DEVICE = None
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current macOS version is not 14.0+ and/or you do not have an MPS-enabled device on this machine.")
    DEVICE = "cpu"
else:
    DEVICE = "mps"
print("Device: ", DEVICE)