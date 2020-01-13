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
from torch.autograd import Variable
from torchvision import transforms, datasets, utils, models
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict 
import seaborn as sb
import time
import json
import argparse



#setting values data loading
#args = parser.parse_args ()
def load_images(data_dir  = './flowers' ):
    
    data_dir = './flowers'
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'



    # TODO: Define your transforms for the training, validation, and testing sets
    # data_transforms
    training_trans = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    testing_trans = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),

                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    validating_trans = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_trans)
    testing_data = datasets.ImageFolder(test_dir ,transform = testing_trans)
    validating_data = datasets.ImageFolder(valid_dir, transform=validating_trans)
    
    #return training_data, testing_data, validating_data
    
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size = 32, shuffle = True)
    validating_loader = torch.utils.data.DataLoader(validating_data, batch_size =32,shuffle = True)
    #data_loaders = [training_loader, testing_loader, validating_loader]
    
    return training_loader, testing_loader, validating_loader


training_data, testing_data, validating_data = load_images('./flowers' )

    #training_loader, testing_loader, validating_loader = load_images('./flowers' )     
   
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
# Load pretrained  vgg network

def settings(arch='vgg13',dropout=0.5, hidden_layers = 120,lr = 0.001, load_gpu='gpu'):
    
    input_size = 25088
   # hidden_layers = 120
    classes = 102
    #dropout=0.5
    lr = 0.001   
    
 #   if arch == 'vgg16':
 #       model = models.vgg16(pretrained=True)
    if arch == 'vgg13':
         model = models.vgg13(pretrained=True)
        
   
 #   else:
 #       print("Others not available, Using vgg16 as default....")
 #       model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
     
        classifier = nn.Sequential(OrderedDict([
             ('dropout',nn.Dropout(dropout)),
             ('inputs', nn.Linear(input_size, hidden_layers)),
             ('relu1', nn.ReLU()),
             ('hidden_layer1', nn.Linear(hidden_layers, 90)),
             ('relu2',nn.ReLU()),
             ('hidden_layer2',nn.Linear(90,80)),
             ('relu3',nn.ReLU()),
             ('hidden_layer3',nn.Linear(80,classes)),
             ('output', nn.LogSoftmax(dim=1))
                                  ]))

        if torch.cuda.is_available() and load_gpu == 'gpu':
            model.cuda()

        model.classifier = classifier

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        model.cuda()

        return model , optimizer ,criterion



     

    


def network_trainer(model, criterion, optimizer, training_loader, validating_loader, epochs =1, print_frequency=20,  load_gpu='gpu'):
    

    steps = 0
    #epochs = 10
    #print_frequency = 6
    loss_show=[]
    
    if torch.cuda.is_available() and load_gpu == 'gpu':
        model.cuda()
        print("Using cuda.......")
    else:    
        model.cpu()
        print("No cuda.......")
#Training the network



   
    starting_time = time.time()

    print('Training in progress...')
    for e in range(epochs):
        
        run_loss = 0
        for ii, (inputs, labels) in enumerate(training_loader):
            steps += 1
           # if torch.cuda.is_available() and load_gpu=='gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()


            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            if steps % print_frequency == 0:
                model.eval()
                validating_lost = 0
                training_accuracy=0


                for ii, (inputs2,labels2) in enumerate(validating_loader):
                    optimizer.zero_grad()

                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        validation_losses = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        training_accuracy += equality.type_as(torch.FloatTensor()).mean()

                validation_losses = validation_losses / len(validating_loader)
                training_accuracy = training_accuracy /len(validating_loader)

                epoch_value = (e+1)/epochs
                losses = run_loss/print_frequency

                time_used = time.time() - starting_time    



        # Displaying Epochs, time etc of training .. process...   
                print("Epoch: {} on {}... ".format(e+1, epochs),
                      "Time used for training:{} ".format(time_used),
                      "Losses: {:.2f}".format(losses),
                      "Validation Losses {:.2f}".format(validation_losses),
                       "Test accuracy: {:.2f}".format(training_accuracy))   

                run_loss = 0
                model.train()
                print('Running ...')
        print('Done ...')
            
def saving_checkpoint(model, path='my_checkpoint.pth',arch ='vgg13', hidden_layers=120,dropout=0.5,lr=0.001,epochs=12):

    model.class_to_idx = training_data.class_to_idx
    model.cpu

    #Save the checkpoint 

    checkpoint = {'model' : model.classifier,
                  'hidden_layers':120,
                  'dropout':dropout,
                  'lr':lr,
                  'nb_of_epochs':epochs,
                  'state_dict':model.state_dict(),
                  'mapping':model.class_to_idx
                 
                 }

    torch.save(checkpoint, 'my_checkpoint.pth')
    
def load_model_checkpoing(checkpoint_path='my_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    loaded_model = models.vgg13()
    hidden_layers = checkpoint['hidden_layers']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model,_,_ = settings(loaded_model , dropout,hidden_layer1,lr)
    model.class_to_idx = checkpoint['mapping']
    model.load_state_dict(checkpoint['state_dict'])
    return model
    
def process_image(images):
  
    # TODO: Process a PIL image for use
    for img in images:
        path = str(img)
    image_pil = Image.open(img)
   
    new_image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    new_image = new_image_transforms(image_pil)
    
    
    return new_image

def predict(image_path, model, topk=5, load_gpu='gpu'):
  #Prediction
    if torch.cuda.is_available() and load_gpu =='gpu':
        model.to('cuda:0')
        print('Using Cuda....')
    tourch_image = process_image(image_path)
    tourch_image = tourch_image.unsqueeze_(0)
    tourch_image = tourch_image.float()
    
    with torch.no_grad():
        sortir = model.forward(tourch_image.cuda())
        
    prob = F.softmax(sortir.data,dim=1)
    
    return prob.topk(topk)
