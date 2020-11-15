#!/usr/bin/env python
# coding: utf-8


import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import torch
cuda = torch.cuda.is_available()

import torch
from torch.autograd import Variable
import torch.optim as optim
from itertools import cycle
import time
from sklearn.manifold import TSNE
import sys
import os
from sklearn.decomposition import PCA

# Custom functions
from dataFunctions import RSNADataset, collate_fn_balanced_batch
from network import CNN_VAE
from lossFunctions import Gaussian_density, kl_a_calc, ELBO_loss, normalize, balanced_accuracy, balanced_binary_cross_entropy

#%%
'''
### Work log ###
Pending tasks:
    - Clean up parameter settings parsing for the network
    - clean up imports 
    - Check what loss functions are either needed or obsolete 
    
Completed tasks:
    - Give data loading it's own document
    - Give the network it's own document
    - Give the loss functions it's own document
'''

#%%
#%% 
'''
Define the global parameters of the network.
'''
img_dimension = [122, 122]
batch_size = 4

num_samples = 5
latent_features = 128
learning_rate = 1e-4

# Define size variables
print_shapes = False
input_channels = 1
num_features = np.prod(img_dimension)*input_channels
## 
classes = [0, 1]

# Regulization
L2_reg = 1e-6
use_dropout = True
do_p_conv = 0.05 # do_p for conv 
do_p_lin = 0.1 # do_p for linear 
batchnorm_eps = 1e-5
batchnorm_momentum = 0.2

## Encoder/Decoder layers:
# Conv/Deconv Layers
conv_out_channels = [32, 32, 64, 64]
conv_kernel =  [5, 5, 3, 3]
conv_padding = [1, 1, 1, 1]
conv_stride =  [1, 1, 1, 1]

# MaxPool Layers
pool_kernel = 2
pool_padding = 0
pool_stride = 2

# Fully connected layers
lin_layer = [500, 200]

## Auxillary network layers:
aux_layer = [500, 200]
aux_variables = 32
aux_in = 2 # layer no. where a is included in encoder
aux_decoder_layers = [100, 200]

## Classifier network layers:
classifier_layer = [500,200,200]
num_classes = len(classes)

# No. of layes
num_conv = len(conv_out_channels)
num_lin = len(lin_layer)
num_aux = len(aux_layer)
num_class = len(classifier_layer)
num_aux_decoder = len(aux_decoder_layers)

# Model path
model_path = 'vae_0.pth'

params = {'latent_features': latent_features,
            'num_samples': num_samples,
            'img_dimension': img_dimension,
            'aux_variables': aux_variables,
            'use_dropout': use_dropout,
            'do_p_lin': do_p_lin,
            'do_p_conv': do_p_conv,
            'batch_size': batch_size,
            'conv_out_channels': conv_out_channels,
            'conv_kernel': conv_kernel,
            'conv_padding': conv_padding,
            'num_classes': num_classes,
            'batchnorm_momentum': batchnorm_momentum,
            'num_conv': num_conv,
            'num_lin': num_lin,
            'num_aux': num_aux,
            'num_aux_decoder': num_aux_decoder,
            'lin_layer': lin_layer,
            'aux_layer': aux_layer,
            'do_p_conv': do_p_conv,
            'conv_stride': conv_stride,
            'aux_decoder_layers': aux_decoder_layers,
            'classifier_layer': classifier_layer,
            'pool_kernel': pool_kernel,
            'pool_padding': pool_padding,
            'pool_stride': pool_stride,
            'input_channels': input_channels,
            'model_path': model_path
          }

#%
'''
Set device
'''
device = torch.device("cuda:0" if cuda else "cpu")
print("Using device:", device)


#%%
'''
Load data (Done)
'''
# Dataloader settings
num_samples_unlabelled = 100
num_samples_labelled = 100
num_samples_test = 100

root_data = r'rsna-pneumonia-detection-challenge'

# Load data with dataloader
print('Loading unlabelled..')
trainDataset_unlabelled = RSNADataset(num_samples=num_samples_unlabelled, img_dimension=img_dimension,
                                      data_path=root_data)
print('\nLoading labelled..')
trainDataset_labelled = RSNADataset(num_samples=num_samples_labelled, num_samples_U=num_samples_unlabelled, 
                                    img_dimension=img_dimension, data_path=root_data)
print('\nLoading test..')
testDataset = RSNADataset(num_samples=num_samples_test, num_samples_U=num_samples_unlabelled, 
                          num_samples_L=num_samples_labelled, img_dimension=img_dimension, data_path=root_data)

# wrap in dataloader
trainloader_labelled = DataLoader(trainDataset_labelled, batch_size=batch_size//2, collate_fn=collate_fn_balanced_batch)
trainloader_unlabelled = DataLoader(trainDataset_unlabelled, batch_size=batch_size//2, collate_fn=collate_fn_balanced_batch)
testloader = DataLoader(testDataset, batch_size=batch_size//2, collate_fn=collate_fn_balanced_batch)

#%%
'''
Define the network and optimizer
'''
# num_samples is the number of samples drawn from the 

net = CNN_VAE(params)
print(net)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_function = ELBO_loss

#%
'''
Test the outputs of the network
'''
x, y = next(iter(trainloader_labelled))
u, _ = next(iter(trainloader_unlabelled))

y_hot =  torch.zeros([batch_size,2])
y_hot = y_hot.scatter_(1,y.type(torch.LongTensor).unsqueeze(1),1)

x, y, y_hot = Variable(x), Variable(y), Variable(y_hot)
if cuda:
    # They need to be on the same device and be synchronized.
    net = net.to(device)
    x, y, y_hot = x.to(device), y.to(device), y_hot.to(device)
    u = u.to(device)
# Run through network with unlabelled data
outputs = net(u)    
elbo_u, elbo_H, elbo_L, kl_u, likelihood_u, kl_u_x, kl_u_a= loss_function(params, u, outputs)
# Run through network with labelled data
outputs = net(x,y_hot)
elbo_l, kl, likelihood_l, kl_l_x, kl_l_a= loss_function(params, x, outputs, y_hot)


# Print dimensions of output 
x_hat = outputs["x_hat"]
mu, log_var = outputs["mu"], outputs["log_var"]
mu_img, log_var_img = outputs["x_mean"], outputs["x_log_var"]
z = outputs["z"]
logits = outputs["y_hat"]
y_hot = y_hot.unsqueeze(dim=1).repeat(1, logits.shape[1], 1)
classification_loss = torch.sum(torch.abs(y_hot - logits))

print('mu:          ', mu.shape, torch.sum(torch.isnan(mu)))
print('log_var:     ', log_var.shape, torch.sum(torch.isnan(log_var)))
print('mu_img:      ', mu_img.shape, torch.sum(torch.isnan(mu_img)))
print('log_var_img: ', log_var_img.shape, torch.sum(torch.isnan(log_var_img)))
print('x:           ', x.shape, torch.sum(torch.isnan(x)))
print('x_hat:       ', x_hat.shape, torch.sum(torch.isnan(x_hat)))
print('z:           ', z.shape, torch.sum(torch.isnan(z)))
print('Total loss:  ', elbo_l)
print('kl:          ', kl)
print('Class. loss: ', classification_loss)
print('Likelihood:  ', likelihood_l)

#%%

# Deleting variable Image and reload package Image
# get_ipython().run_line_magic('reset_selective', '-f "^Image$"')

num_epochs = 301 # No_train_samples // batch_size
batch_epo_u = num_samples_unlabelled // batch_size
batch_epo_l = num_samples_labelled // batch_size
batch_per_epoch = batch_epo_u if batch_epo_u>batch_epo_l else batch_epo_l


tmp_img = "tmp_vae_out.png"
show_sampling_points = False
classes = [0, 1]

train_loss, valid_loss = [], []
train_kl, valid_kl = [], []
train_likelihood, valid_likelihood = [], []
train_classification, valid_classification = [], []
train_acc, valid_acc = [], []
train_L, train_H, train_kl_u_x, train_kl_u_a, train_elbo_l, train_kl_l_x, train_kl_l_a = [], [], [], [], [], [], []
valid_L, valid_H, valid_kl_u_x, valid_kl_u_a, valid_elbo_l, valid_kl_l_x, valid_kl_l_a = [], [], [], [], [], [], []
# train_latent_loss, valid_latent_loss = [], []
total_loss = []

# Classification Loss
alpha = 10 * (num_samples_unlabelled + num_samples_labelled) / num_samples_labelled   
print("Weight of classification loss (alpha): ", alpha)

# Latent Loss
beta = 0   

# Deterministic Warm-Up
deterministic_increment = 75  # The increment in the deterministic warmup. n = 10 adds 1/10 to kl_warmup for every epoch.
t_max = 1 # Maximal weight of KL divergence


for epoch in range(num_epochs):
    batch_loss, batch_kl, batch_likelihood, batch_acc, batch_classification = [], [], [], [], []
    batch_L, batch_H, batch_kl_u_x, batch_kl_u_a, batch_elbo_l, batch_kl_l_x, batch_kl_l_a = [], [], [], [], [], [], []
#     batch_latent_loss = []
    net.train()
    
    # Deterministic Warmup for KL divergence
    kl_warmup = 1
    '''if epoch >= deterministic_increment:
        kl_warmup = 1
    else:
        kl_warmup = epoch * 1/deterministic_increment       
    print("KL Warm-Up: ", kl_warmup)'''
    ###############################
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    count = 0
    total_loss_batch = 0
    for (x, y), (u, _) in zip(cycle(trainloader_labelled), trainloader_unlabelled):
        y_hot = torch.zeros([batch_size, 2])
        y_hot = y_hot.scatter_(1, y.type(torch.LongTensor).unsqueeze(1), 1)
        x, y, u, y_hot = Variable(x), Variable(y), Variable(u), Variable(y_hot)
        if cuda:
            # They need to be on the same device and be synchronized.
            x, y, y_hot = x.to(device), y.to(device), y_hot.to(device)
            u = u.to(device)

        count = count + 1
        if not count % (batch_per_epoch/4):
            print("Epoch:", epoch, "Batch:", count, "/", batch_per_epoch)

        #### Unlabelled
        outputs = net(u)
        elbo_u, elbo_H, elbo_L, kl_u, likelihood_u, kl_u_x, kl_u_a = loss_function(params, u, outputs, None,kl_warmup)
        elbo_H = elbo_H.cpu().detach().numpy()
        elbo_L = elbo_L.cpu().detach().numpy()
        kl_u_x = kl_u_x.cpu().detach().numpy()
        kl_u_a = kl_u_a.cpu().detach().numpy()
        
        #### Labelled
        outputs = net(x,y_hot)
        logits = outputs["y_hat"]
        y_hot = y_hot.unsqueeze(dim=1).repeat(1, logits.shape[1], 1)
        
        # Weight the classification loss for the two classes
        classification_loss = alpha * balanced_binary_cross_entropy(logits, y_hot)
        acc = balanced_accuracy(logits, y_hot)
        batch_acc.append(100*acc)
        elbo_l, kl_l, likelihood_l, kl_l_x, kl_l_a = loss_function(params, x, outputs, y_hot,kl_warmup)
        kl_l_x = kl_l_x.cpu().detach().numpy()
        kl_l_a = kl_l_a.cpu().detach().numpy()
        
        # Latent lose
#         rho = torch.tensor(0.05)
#         act = outputs["activation"]
#         act = (act-torch.min(act)) / torch.max(act-torch.min(act))
#         # Mean over batches (training set): from [batch_size, num_samples, num_hidden_units] -> [num_samples, num_hidden_units]
#         rho_hat = torch.mean(torch.abs(act), dim = 0)
#         # Sum over num_samples and num_hidden_units
#         latent_loss = beta * torch.torch.sum(torch.abs(rho * torch.log( rho / rho_hat ) 
#                                           + (1 - rho) * torch.log( (1-rho) / (1-rho_hat)))).to(device)

        
        # Combine losses
        loss =  elbo_l + elbo_u + classification_loss #+ latent_loss # notice alpha has been moved to where classification loss is calculated
        total_loss_batch += loss.item()
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        elbo_l = elbo_l.cpu().detach().numpy()
        batch_classification.append(classification_loss.item())
        batch_loss.append(loss.item())
        batch_kl.append(kl_u.item()+kl_l.item())
        batch_likelihood.append(likelihood_u.item() + likelihood_l.item())
        batch_L.append(elbo_L)  # NOT elbo_l, but L from elbo_u
        batch_H.append(elbo_H)
        batch_kl_u_x.append(kl_u_x)
        batch_kl_u_a.append(kl_u_a)
        batch_elbo_l.append(elbo_l)
        batch_kl_l_x.append(kl_l_x)
        batch_kl_l_a.append(kl_l_a)
#         latent_loss = latent_loss.cpu().detach().numpy()
#         batch_latent_loss.append(latent_loss)
    
    train_classification.append(np.mean(batch_classification))
    train_acc.append(np.mean(batch_acc))    
    train_loss.append(np.mean(batch_loss))
    train_kl.append(np.mean(batch_kl))
    train_likelihood.append(np.mean(batch_likelihood))
    total_loss.append(total_loss_batch / (batch_size*batch_per_epoch))
    train_L.append(np.mean(batch_L))
    train_H.append(np.mean(batch_H))
    train_kl_u_x.append(np.mean(batch_kl_u_x))
    train_kl_u_a.append(np.mean(batch_kl_u_a))
    train_elbo_l.append(np.mean(batch_elbo_l))
    train_kl_l_x.append(np.mean(batch_kl_l_x))
    train_kl_l_a.append(np.mean(batch_kl_l_a))
#     train_latent_loss.append(np.mean(batch_latent_loss))

    # Evaluate, do not propagate gradients
    with torch.no_grad():
        net.eval()
        
        # Just load a single batch from the test loader
        x, y = next(iter(testloader))
        y_hot = torch.zeros([batch_size,2])
        y_hot = y_hot.scatter_(1,y.type(torch.LongTensor).unsqueeze(1),1)
        x, y, y_hot = Variable(x), Variable(y), Variable(y_hot)
        if cuda:
            # They need to be on the same device and be synchronized.
            x, y_hot = x.to(device), y_hot.to(device)
        
        # Unlabelled
        outputs = net(u)
        elbo_u, elbo_H, elbo_L, kl_u, likelihood_u, kl_u_x, kl_u_a = loss_function(params, u, outputs, None, kl_warmup)
        elbo_H = elbo_H.cpu().detach().numpy()
        elbo_L = elbo_L.cpu().detach().numpy()
        kl_u_x = kl_u_x.cpu().detach().numpy()
        kl_u_a = kl_u_a.cpu().detach().numpy()
        
        # Labelled
        outputs = net(x,y_hot)
        x_hat = outputs['x_hat']
        logits = outputs["y_hat"]
        z = outputs["z"]
        
        y_hot = y_hot.unsqueeze(dim = 1).repeat(1,logits.shape[1],1)
        # acc = torch.sum(y_hot.view(-1,2) * logits.view(-1,2), dim = 1).cpu().detach().numpy()
        # valid_acc.append(sum(acc > 0.5) / batch_size*num_samples)
        acc = balanced_accuracy( logits, y_hot)
        valid_acc.append(100*acc)
        
        # Weight the classification loss for the two classes
        # classWeight = torch.FloatTensor([torch.sum(y_hot[:,1,0])/torch.sum(y_hot[:,1,]), torch.sum(y_hot[:,1,1])/torch.sum(y_hot[:,1,])]).to(device)
        # classification_loss = alpha*torch.nn.functional.binary_cross_entropy(logits,y_hot, weight=classWeight)
        # classification_loss = torch.sum(torch.abs(y_hot - logits))
        # classification_loss = alpha*torch.nn.functional.binary_cross_entropy(logits,y_hot)
        classification_loss = alpha * balanced_binary_cross_entropy(logits, y_hot)
        # elbo, kl = loss_function(x_hat, x, mu, log_var)
        elbo_l, kl_l, likelihood_l, kl_l_x, kl_l_a = loss_function(params, x, outputs, kl_warmup)
        kl_l_x = kl_l_x.cpu().detach().numpy()
        kl_l_a = kl_l_a.cpu().detach().numpy()
        
        # We save the latent variable and reconstruction for later use
        # we will need them on the CPU to plot
        x = x.to("cpu")
        x_hat = x_hat.to("cpu")
        z = z.detach().to("cpu").numpy()

        # Append to lists
        valid_classification.append(classification_loss.item())
        valid_loss.append(elbo_l.item()+classification_loss.item()+elbo_u.item())
        valid_kl.append(kl_l.item())
        valid_likelihood.append(likelihood_l.item())
        valid_L.append(elbo_L)
        valid_H.append(elbo_H)
        valid_kl_u_x.append(kl_u_x)
        valid_kl_u_a.append(kl_u_a)
        valid_elbo_l.append(elbo_l.item())
        valid_kl_l_x.append(kl_l_x)
        valid_kl_l_a.append(kl_l_a)

    # Save model every epoch
    print("INFO: Saving the model at ", model_path)
    torch.save(net.state_dict, model_path)

    if epoch == 0:
        continue
    
    if epoch % 10:
        continue
    
    validation_from = 20
    if epoch < validation_from:
        VALIDATION = False
    else:
        VALIDATION = True
    
    # -- Plotting --
    f, axarr = plt.subplots(5, 2, figsize=(20, 30))

    # Loss
    ax = axarr[0, 0]
    ax.set_title("ELBO")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')

    ax.plot(np.arange(epoch+1), train_loss, color="black")
    if VALIDATION:
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_loss[validation_from:len(valid_loss)], color="gray", linestyle="--")
        ax.legend(['Training', 'Validation'])
    else:
        ax.legend(['Training'])

    # Latent space
    ax = axarr[0, 1]
    ax.set_title('Latent space')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    if batch_size > 4:
        rows = 4
        columns = batch_size // rows if batch_size // rows < 8 else 8            
    else:
        rows = 2
        columns = 2
    
    span = np.linspace(-4, 4, rows)
    grid = np.dstack(np.meshgrid(span, span)).reshape(-1, 2)
    
    # If you want to use a dimensionality reduction method you can use
    # for example PCA by projecting on two principal dimensions

    """ PCA """
    # z = PCA(n_components=2).fit_transform(z.reshape(-1,latent_features))
    # z = z.reshape(batch_size,num_samples,2)
    
    """ TSNE """
    print(z.shape)
    z = TSNE(n_components=2).fit_transform(z.reshape(-1,latent_features))
    z = z.reshape(batch_size,num_samples**2,2)
    # z = z[:,1,] # We do only want to plot one z instead of |num_samples|
    
    colors = iter(plt.get_cmap('Set1')(np.linspace(0, 1.0, len(classes))))
    for c in classes:
        ax.scatter(*z[c == y.numpy()].reshape(-1, 2).T, c=next(colors), marker='o', label=c)
        
    if show_sampling_points:
        ax.scatter(*grid.T, color="k", marker="x", alpha=0.5, label="Sampling points")

    ax.legend()
    
    # KL / reconstruction
    ax = axarr[1, 0]
    ax.set_title("Losses")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title("ELBO: Labelled and unlabelled")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')        
    if VALIDATION:
        ax.plot(np.arange(epoch+1), train_elbo_l, color="red")
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_elbo_l[validation_from:len(valid_loss)], color="red", linestyle="--")
        ax.plot(np.arange(epoch+1), train_L, color="green")  
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_L[validation_from:len(valid_loss)], color="green", linestyle="--") 
        ax.legend(['Train. ELBO Labelled',  'Valid. ELBO Labelled', 'Train. ELBO Unlabelled', 'Valid. ELBO Unlabelled'])
    else:
        ax.plot(np.arange(epoch+1), train_elbo_l, color="red")
        ax.plot(np.arange(epoch+1), train_L, color="green")    # Typically L >>> H, therefore not L+H   
        ax.legend(['Train. ELBO Labelled', 'Train. ELBO Unlabelled'])
    
    # Latent space samples
    ax = axarr[1, 1]
    ax.set_title('Samples from latent space')
    ax.axis('off')

    with torch.no_grad():
        epsilon = torch.randn(batch_size, latent_features).to(device)
        samples = torch.sigmoid(net.sample_from_latent(epsilon)).cpu().detach().numpy()
    print(samples.shape)

    canvas = np.zeros((img_dimension[0]*rows, img_dimension[1]*columns))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            #temp_img = samples[idx].reshape((224, 224))
            #canvas[i*28:(i+1)*28, j*28:(j+1)*28] = resize(temp_img, output_shape=[28,28], mode='reflect', anti_aliasing=True)
            canvas[i*img_dimension[0]:(i+1)*img_dimension[0], j*img_dimension[1]:(j+1)*img_dimension[1]] = \
                samples[idx,0:np.prod(img_dimension)].reshape(img_dimension)
    ax.imshow(canvas, cmap='gray')

    # Input images
    ax = axarr[2, 0]
    ax.set_title('Inputs')
    ax.axis('off')

    canvas = np.zeros((img_dimension[0]*rows, img_dimension[1]*columns))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            #temp_img = x[idx].reshape((224, 224))
            #canvas[i*28:(i+1)*28, j*28:(j+1)*28] = resize(temp_img, output_shape=[28,28], mode='reflect', anti_aliasing=True)
            canvas[i*img_dimension[0]:(i+1)*img_dimension[0], j*img_dimension[1]:(j+1)*img_dimension[1]] = \
                x[idx,0:img_dimension[0]*img_dimension[1]].reshape(img_dimension)
    ax.imshow(canvas, cmap='gray')

    # Reconstructions
    ax = axarr[2, 1]
    ax.set_title('Reconstructions: mean')
    ax.axis('off')

    canvas = np.zeros((img_dimension[0]*rows, img_dimension[1]*columns))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i*img_dimension[0]:(i+1)*img_dimension[0], j*img_dimension[1]:(j+1)*img_dimension[1]] = \
                normalize(x_hat[idx,0:img_dimension[0]*img_dimension[1]].reshape(img_dimension))
    ax.imshow(canvas, cmap='gray')

    # Latent features
    ax = axarr[3, 0]
    ax.axis('off')
    ax.set_title('Displaying latent features')
    with torch.no_grad():
        latent_feature = torch.zeros(batch_size, latent_features).to(device)
        for i in range(0, latent_features if latent_features <= batch_size else batch_size):
            latent_feature[i, i] = 100
        displaying_latent_features = (net.sample_from_latent(latent_feature)).cpu().detach() #no sigmoid, instead normalize later
    
    # Determine number of rows and columns based on the size of latent space
    if displaying_latent_features.shape[0] == 64: 
        rows_2 = 8
        columns_2 = 8
    elif displaying_latent_features.shape[0] == 128: 
        rows_2 = 8
        columns_2 = 16
    else:
        rows_2 = rows
        columns_2 = columns   
    canvas = np.zeros((img_dimension[0]*rows_2, img_dimension[1]*columns_2))  
    for i in range(rows_2):
        for j in range(columns_2):
            idx = i % columns_2 + rows_2 * j
            canvas[i*img_dimension[0]:(i+1)*img_dimension[0], j*img_dimension[1]:(j+1)*img_dimension[1]] = \
                normalize(displaying_latent_features[idx,0:img_dimension[0]*img_dimension[1]].reshape(img_dimension))
    ax.imshow(canvas, cmap='gray')

    # Classification Loss
    ax = axarr[3, 1]
    ax.set_title("Classification Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    
    ax.plot(np.arange(epoch+1), train_acc, color="black")
    ax.plot(np.arange(epoch+1), valid_acc, color="gray", linestyle="--")
    ax.legend(['Training', 'Validation'])
    
    # KL Divergence - Unlabelled
    ax = axarr[4,0]
    ax.set_title("KL Divergence - Unlabelled")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if VALIDATION:
        ax.plot(np.arange(epoch+1), train_kl_u_x, color="blue")
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_kl_u_x[validation_from:len(valid_loss)], color="blue", linestyle="--")
        ax.plot(np.arange(epoch+1), train_kl_u_a, color="orange")
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_kl_u_a[validation_from:len(valid_loss)], color="orange", linestyle="--")
        ax.legend(['Train. kl_u_x',  'Valid. kl_u_x', 'Train. kl_u_a', 'Valid. kl_u_a'])
    else:
        ax.plot(np.arange(epoch+1), train_kl_u_x, color="blue")
        ax.plot(np.arange(epoch+1), train_kl_u_a, color="orange")
        ax.legend(['Train. kl_u_x',  'Train. kl_u_a'])

    # KL Divergence - Labelled
    ax = axarr[4,1]
    ax.set_title("KL Divergence - Labelled")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if VALIDATION:
        ax.plot(np.arange(epoch+1), train_kl_l_x, color="blue")
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_kl_l_x[validation_from:len(valid_loss)], color="blue", linestyle="--")
        ax.plot(np.arange(epoch+1), train_kl_l_a, color="orange")
        ax.plot(np.arange(epoch+1)[validation_from:len(valid_loss)], valid_kl_l_a[validation_from:len(valid_loss)], color="orange", linestyle="--")
        # ax.plot(np.arange(epoch+1), train_latent_loss, color="green")
        ax.legend(['Train. kl_l_x',   'Valid. kl_l_x', 'Train. kl_l_a', 'Valid. kl_l_a'])
        # ax.legend(['Train. kl_l_x',   'Valid. kl_l_x', 'Train. kl_l_a', 'Valid. kl_l_a', 'Latent_loss'])
    else:
        ax.plot(np.arange(epoch+1), train_kl_l_x, color="blue")
        ax.plot(np.arange(epoch+1), train_kl_l_a, color="orange")
        # ax.plot(np.arange(epoch+1), train_latent_loss, color="green")
        ax.legend(['Train. kl_l_x',  'Train. kl_l_a'])
        # ax.legend(['Train. kl_l_x',  'Train. kl_l_a',  'Latent_loss'])
    
    plt.savefig(tmp_img)
    plt.close(f)

    os.remove(tmp_img)
    
    if epoch % 20:
        continue
    
    # Run trough full test data set
    TP, FP, FN, TN, P, N = 0, 0, 0, 0, 0, 0
    for (x, y) in testloader:
        y_hot =  torch.zeros([batch_size,2])

        for i in range(len(y)):
            y_hot[i] = torch.tensor([0, 1]) if y[i]==1 else torch.tensor([1, 0])
        x, y, y_hot = Variable(x), Variable(y), Variable(y_hot)
        if cuda:
            # They need to be on the same device and be synchronized.
            x, y_hot = x.to(device), y_hot.to(device)

        #### Labelled
        outputs = net(x,y_hot)
        logits = outputs["y_hat"]
        y_hot = y_hot.unsqueeze(dim = 1).repeat(1,logits.shape[1],1)
        acc_1, TP_1, FP_1, FN_1, TN_1, P_1, N_1 = balanced_accuracy(logits, y_hot, return_all = True)
        TP += TP_1
        FP += FP_1
        FN += FN_1
        TN += TN_1
        P  += P_1
        N  += N_1

    acc =  torch.sum(TP/P + TN/N)/2  
    print("Acc: ",acc)    
    print("TP: ",TP)
    print("FP: ",FP)
    print("TN: ",TN)
    print("FN: ",FN)
    print("P: ",P)
    print("N: ",N)

#%%
'''
# Test Encoder
encoder = Encoder()
print(encoder)
x = torch.randn(batch_size,1,width, height)
x = Variable(x)
if cuda:
    encoder = encoder.cuda()
    x = x.cuda()
    y = None
y = encoder(x)
print(y.shape)
del y,x

                                           
# Test decoder
layer_size = get_layer_sizes(width,height,conv_out_channels,conv_kernel,conv_padding,conv_stride)
decoder = Decoder(layer_size)
print(decoder)
z = torch.randn(batch_size,num_samples,latent_features)
y = torch.randn(batch_size,num_samples,2)
z = Variable(z)
y = Variable(y)
if cuda:
    decoder = decoder.cuda()
    z = z.cuda()
    y = y.cuda()
x_hat = decoder(z,y)
print(x_hat.shape)
del y,z

# Test Classifier
classifier = Classifier()
print(classifier)
xa = torch.randn(batch_size,1,lin_layer[-1]+aux_variables)
xa = Variable(xa)
if cuda:
    classifier = classifier.cuda()
    xa = xa.cuda()
y = classifier(xa)
print(y.shape)
                                           
                                           
net = CNN_VAE(latent_features, num_samples)
print(net)
x = torch.randn(batch_size,1,width, height)
x = Variable(x)
y = None
if cuda:
    net = net.cuda()
    x = x.cuda()

outputs = net(x)
x_hat = outputs["x_hat"]
mu, log_var = outputs["mu"], outputs["log_var"]
mu_img, log_var_img = outputs["x_mean"], outputs["x_log_var"]
z = outputs["z"]
logits = outputs["y_hat"]
a_mu = outputs["p_a_mu"]
a_log_var = outputs["p_a_log_var"]
q_a_mu = outputs["q_a_mu"]
q_a_log_var = outputs["q_a_log_var"]
p_a_mu = outputs["p_a_mu"]
p_a_log_var = outputs["p_a_log_var"]

print('mu:          ',mu.shape,torch.sum(torch.isnan(mu)))
print('log_var:     ',log_var.shape,torch.sum(torch.isnan(log_var)))
print('mu_img:      ',mu_img.shape,torch.sum(torch.isnan(mu_img)))
print('log_var_img: ',log_var_img.shape,torch.sum(torch.isnan(log_var_img)))
print('x:           ',x.shape,torch.sum(torch.isnan(x)))
print('x_hat:       ',x_hat.shape,torch.sum(torch.isnan(x_hat)))
print('z:           ',z.shape,torch.sum(torch.isnan(z)))
print('y_hat:       ',logits.shape,torch.sum(torch.isnan(logits)))
print('a_mu:        ',a_mu.shape,torch.sum(torch.isnan(a_mu)))
print('a_log_var:   ',a_log_var.shape,torch.sum(torch.isnan(a_log_var)))
print('q_a_mu:      ',q_a_mu.shape,torch.sum(torch.isnan(q_a_mu)))
print('q_a_log_var: ',q_a_log_var.shape,torch.sum(torch.isnan(q_a_log_var)))
print('p_a_mu:      ',p_a_mu.shape,torch.sum(torch.isnan(p_a_mu)))
print('p_a_log_var: ',p_a_log_var.shape,torch.sum(torch.isnan(p_a_log_var)))
del y,x
# Garbage collector...
#import gc
#gc.collect()

#for parameter in net.parameters():
#    print(parameter.shape)
#epsilon = torch.randn(batch_size, latent_features).cuda
#samples = torch.sigmoid(net.decoder(net.latent_to_CNN(epsilon))).detach()
'''
