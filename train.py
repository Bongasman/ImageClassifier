# Imports here
import numpy as np
import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms, datasets, utils, models
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict 
from torch.autograd import Variable
import seaborn as sb
import time
import json
import tensorflow as tf
import argparse
import classifier_functions

inputs = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

inputs.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
inputs.add_argument('--gpu', dest="gpu", action="store", default="gpu")
inputs.add_argument('--save_dir', dest="save_dir", action="store", default="./my_checkpoint.pth")
inputs.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
inputs.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
inputs.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
inputs.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str)
inputs.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)


parse = inputs.parse_args()
lr = parse.learning_rate
data_files = parse.data_dir
dropout = parse.dropout
hidden_layers = parse.hidden_units
path = parse.save_dir
arch = parse.arch
load_gpu = parse.gpu
epochs = parse.epochs
#super().__init__(DataLoader)
def main():
   

    training_data, testing_data, validating_data = classifier_functions.load_images('./flowers/')

    training_loader, testing_loader, validating_loader  = classifier_functions.load_images('./flowers/')
        
    model, optimizer, criterion = classifier_functions.settings(arch,dropout,hidden_layers,lr,load_gpu)


    classifier_functions.network_trainer(model, criterion, optimizer,  training_loader, validating_loader, epochs, 20, load_gpu)


    classifier_functions.saving_checkpoint(model, path,arch,hidden_layers,dropout,lr)

    print('Finalising training.....................................Done !')
if __name__== "__main__":
    main()