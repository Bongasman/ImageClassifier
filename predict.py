# Imports here
import numpy as np
import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import datasets,  models, transforms
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict 
import seaborn as sb
import time
import json
import tensorflow as tf
import argparse

import classifier_functions

#Command Line Arguments

inputs = argparse.ArgumentParser(description='Prediction')
inputs.add_argument('input_img', default='./flowers/test/10/image_07104.jpg', nargs='*', action="store", type = str)
inputs.add_argument('checkpoint', default='./my_checkpoint.pth', nargs='*', action="store",type = str)
inputs.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
inputs.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
inputs.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parse = inputs.parse_args()
path_image = parse.input_img
number_of_outputs = parse.top_k
load_gpu = parse.gpu
#input_img = parse.input_img
checkpoint_path = parse.checkpoint



#training_loader, testing_loader, validating_loader = classifier_functions.load_data()


model=classifier_functions.load_model_checkpoint(checkpoint_path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


proba = classifier_functions.predict(path_image, model, number_of_outputs, load_gpu)


labels = [cat_to_name[str(index + 1)] for index in np.array(proba[1][0])]
proba = np.array(proba[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], proba[i]))
    i += 1

print("Prediction successful !")