import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r"C:\Users\Áron\Desktop\Courses\csgo-ai-competition-master\dataset_initial")

t = torch.tensor([1,23,3])
