# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.nn import Linear, GRU, Conv2d, Dropout, Dropout2d, MaxPool2d, BatchNorm1d, BatchNorm2d, ReLU, ELU, ConvTranspose2d, MaxUnpool2d, Softmax, Sigmoid, Softsign
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax, dropout, dropout2d, interpolate, softplus
import torch.nn as nn
from torch.autograd import Variable
cuda = torch.cuda.is_available()

# img_dimension = [122, 122]
# batch_size = 2


# num_samples = 5
# latent_features = 128
# learning_rate = 1e-4

# # Define size variables
# print_shapes = False
# height = img_dimension[0]  
# width = img_dimension[1]
# IMG_SIZE = width
# channels = 1
# num_features = height*width*channels
# ## 
# classes = [0, 1]

# # Regulization
# L2_reg = 1e-6
# use_dropout = True
# do_p_conv = 0.05 # do_p for conv 
# do_p_lin = 0.1 # do_p for linear 
# batchnorm_eps = 1e-5
# batchnorm_momentum = 0.2

# ## Encoder/Decoder layers:
# # Conv/Deconv Layers
# conv_out_channels = [32, 32, 64, 64]
# conv_kernel =  [5, 5, 3, 3]
# conv_padding = [1, 0, 1, 1]
# conv_stride =  [2, 1, 1, 1]

# # MaxPool Layers
# pool_kernel = 2
# pool_padding = 0
# pool_stride = 2

# # Fully connected layers
# lin_layer = [500, 200]

# ## Auxillary network layers:
# aux_layer = [500, 200]
# aux_variables = 32
# aux_in = 2 # layer no. where a is included in encoder
# aux_decoder_layers = [100, 200]

# ## Classifier network layers:
# classifier_layer = [500,200,200]
# num_classes = len(classes)

# # No. of layes
# num_conv = len(conv_out_channels)
# num_lin = len(lin_layer)
# num_aux = len(aux_layer)
# num_class = len(classifier_layer)
# num_aux_decoder = len(aux_decoder_layers)

'''
Network
'''       
class CNN_VAE(nn.Module):
    def __init__(self, params):
        super(CNN_VAE, self).__init__(), 
        
        self.params = params
        
        ## Networks
        self.layer_size = get_layer_sizes(params['img_dimension'], params['conv_out_channels'], 
                                          params['conv_kernel'], params['conv_padding'], params['conv_stride'],
                                          params['pool_kernel'], params['pool_padding'], params['pool_stride'])
        
        
        # self.encoder = Encoder(height=params['img_dimension'][0], width=params['img_dimension'][1], conv_out_channels=params['conv_out_channels'], 
        #          num_conv=params['num_conv'], input_channels=params['input_channels'], conv_kernel=params['conv_kernel'], 
        #          conv_stride=params['conv_stride'], conv_padding=params['conv_padding'], batchnorm_momentum=params['batchnorm_momentum'], 
        #          pool_kernel=params['pool_kernel'], pool_stride=params['pool_stride'], pool_padding=params['pool_padding'], 
        #          num_lin=params['num_lin'], lin_layer=params['lin_layer'], do_p_lin=params['do_p_lin'], use_dropout=params['use_dropout'],
        #          batch_size=params['batch_size'])
        self.encoder = Encoder(params)
        self.decoder = Decoder(params, self.layer_size)
        self.classifier = Classifier(params)
        self.aux_encoder = Aux_encoder(params)
        self.aux_decoder = Aux_decoder(params)
        
        # map to latent space
        Additional_layer = nn.ModuleList()
        Additional_layer.append(Linear(in_features=params['lin_layer'][-1]+params['aux_variables']+
                                       params['num_classes'], out_features=params['latent_features']*2))
        Additional_layer.append(BatchNorm1d(params['latent_features']*2, momentum = params['batchnorm_momentum']))
        self.add_module("Additional_layer",Additional_layer)
       
        # Initialize weight of layers
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data = torch.tensor(0.01)
                
    def forward(self, x, y=None):
        outputs = {}
        self.indices = []
        self.layer_size = []
        
        ## Run through encoder:
        x = self.encoder(x);
        
        ## Run through auxillary encoder
        if self.aux_variables > 0:
            q_a_mu, q_a_log_var = self.aux_encoder(x)
            q_a = gaussian_sample(q_a_mu,q_a_log_var,self.params['num_samples'],self.params['aux_variables']) # sample auxillary variables
            xa = torch.cat([x.unsqueeze(1).repeat(1,self.params['num_samples'],1),q_a],dim=2) # Create combined vector of x and q_a
        else:
            xa = x.unsqueeze(1)
        
        if cuda:
            xa = xa.cuda()
        ## Run trough classifier
        logits = self.classifier(xa)
        
        ## Map x, y, a to latent space
        lat_in = self.run_module(self.map_to_latent_space,xa,y)
        if y is None:
            # marginalize over y sampling.
            lat_in = marginalizeY(lat_in,self.num_classes,1)
        # Split into mu and log_var
        mu, log_var = torch.chunk(lat_in, 2, dim=-1)
        # Make sure that the log variance is positive
        log_var = softplus(log_var)
        ## Sample from latent space
        z = gaussian_sample(mu,log_var, self.params['num_samples']*mu.size(1), self.params['latent_features'])
        
        ## aux. decoder
        if self.aux_variables > 0:
            xz = torch.cat([z, x.unsqueeze(1).repeat(1,z.shape[1],1)], dim = -1)
            ## Expand to be able to sample from y
            a = self.run_module(self.aux_decoder,xz,y)
            #a_mean, a_log_var = torch.chunk(a, 2, dim=-1) # the mean and log_var reconstructions from the decoder
            a_hat, a_log_var, a_mean = output_recon(a)
            a_log_var = softplus(a_log_var)    
    
        ## Decoder
        x = self.run_module(self.decoder,z,y)
        x_hat, x_log_var, x_mean = output_recon(x)
        
        ## Assign variables
        outputs["x_hat"] = x_hat # This is used for visulizations only 
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["log_var"] = log_var
        
        # image recontructions (notice they are outputted as matrices)
        outputs["x_mean"] =  x_mean 
        outputs["x_log_var"] = x_log_var 
        
        # auxillary outputs
        if self.aux_variables > 0:      
            outputs["q_a_mu"] = q_a_mu
            outputs["q_a_log_var"] = q_a_log_var
            outputs["p_a_mu"] = a_mean
            outputs["p_a_log_var"] = a_log_var
            outputs["q_a"] = q_a
            
        # classifier outputs 
        outputs["y_hat"] = logits

        return outputs
    
    def map_to_latent_space(self,xa,y):
        z = torch.cat([xa,y],dim=-1)
        if cuda:
            z = z.cuda()
        z = self.Additional_layer[0](z)
        z = z.permute(0,2,1)
        z = self.Additional_layer[1](z)
        z = z.permute(0,2,1)
        z = relu(z)
        if self.params['use_dropout']:
            z = dropout(z, p=self.params['do_p_lin']) 
        return z
    
    
    def sample_y(self,batch_size,num_samples,num_classes,i):
        tmp = Variable(torch.zeros(num_classes))
        tmp[i] = 1
        tmp = tmp.cuda()
        return tmp.repeat(batch_size,num_samples,1)
    
    def sample_from_latent(self, x):
        x_UL = []
        for j in range(self.num_classes):
            tmp = self.decoder(x.unsqueeze(1).repeat(1,self.num_samples,1), 
                               self.sample_y(self.params['batch_size'], self.params['num_samples'], 
                                             self.params['num_classes'], j))
            x_UL.append(tmp)
        x_hat, _, _ = output_recon(sum(x_UL))
        return x_hat
    
    def onehotEncode(self,num_classes,num_samples,batch_size):
        # Define labels as (0, ... , num_classes)
        labels = torch.Tensor(range(num_classes)).type(torch.LongTensor)
        # Tile labels to get it repeatet tile wise
        labels = labels.unsqueeze(1).repeat(1,num_samples).view(-1,1).squeeze(0)
        # initialize y matrix
        y = torch.zeros(labels.size(0), num_classes)
        # onehot encode labels
        y.scatter_(1, labels, 1)
        # add batchsize dimension
        y = y.unsqueeze(0).repeat(batch_size,1,1)
        return (y)
    
    def run_module(self,module,x, y = None):
        #wrapper function for running a module either labeled or unlabeled
        
        if y is None: # If running unlabeled inputs
            y = self.onehotEncode(self.params['num_classes'], x.size(1), self.params['batch_size'])
            if cuda:
                y = y.cuda()
            x = x.repeat(1,self.params['num_classes'],1)
        else:
            y = y.unsqueeze(1).repeat(1,x.size(1),1)
        a = module(x,y)
        return a    
        
 
''' 
Encoders
'''
class Encoder(nn.Module): 
    def __init__(self, params):
        super(Encoder, self).__init__()
        
        self.params = params
        
        # Dropout params
        self.do_p_lin = params['do_p_lin']
        self.use_dropout = params['use_dropout']
        self.batch_size = params['batch_size']
        
        
        # height, width, last_num_channels, num_layers, conv_kernel, conv_padding, 
        #                     conv_stride, pool_kernel, pool_padding, pool_stride
        # Calculate final size of the CNN
        self.final_dim = compute_final_dimension(height=params['img_dimension'][0], 
                            width=params['img_dimension'][1], last_num_channels=params['conv_out_channels'][-1],
                            num_conv=params['num_conv'],
                            conv_kernel=params['conv_kernel'], conv_padding=params['conv_padding'], 
                            conv_stride=params['conv_stride'], pool_kernel=params['pool_kernel'], 
                            pool_padding=params['pool_padding'], pool_stride=params['pool_stride'])
        
        in_channels = params['input_channels']
        ## Convolutional layers of the encoder
        Encoder_conv = nn.ModuleList()
        for i in range(params['num_conv']):
            Encoder_conv.append(Conv2d( in_channels=in_channels,
                                            out_channels=params['conv_out_channels'][i],
                                            kernel_size=params['conv_kernel'][i],
                                            stride=params['conv_stride'][i],
                                            padding=params['conv_padding'][i]))
            Encoder_conv.append(BatchNorm2d(params['conv_out_channels'][i]))
            Encoder_conv.append(MaxPool2d(  kernel_size=params['pool_kernel'], 
                                        stride=params['pool_stride'],
                                        padding=params['pool_padding'],
                                        return_indices = False))
            in_channels = params['conv_out_channels'][i]
        self.add_module("Encoder_conv",Encoder_conv)
        
        # Fully connected layers of encoder
        Encoder_FC = nn.ModuleList()
        in_weights = self.final_dim[0]
        for i in range(params['num_lin']):
            Encoder_FC.append(Linear(in_features=in_weights, out_features=params['lin_layer'][i]))
            Encoder_FC.append(BatchNorm1d(params['lin_layer'][i]))
            in_weights = params['lin_layer'][i]
        self.add_module("Encoder_FC",Encoder_FC)
        
    def forward(self,x):
        '''
        Inputs:
        x - Input image [batch_size, channels, width, height]
        Outputs:
        x - output from encoders conv + lin layers. [batch_size, num_features]
        '''
        # Convolutional layers of encoder
        for i in range(0,len(self.Encoder_conv),3):
            x = self.Encoder_conv[i](x) # Convolutional layer
            x = self.Encoder_conv[i+1](x) # Batchnorm layer
            x = relu(x)
            if self.use_dropout:
                x = dropout2d(x, p=self.do_p_conv)   
            x = self.Encoder_conv[i+2](x) # Maxpool Layer
        x = x.view(self.batch_size, -1) # Prepare x for linear layers
        
        # Fully connected layers of encoder
        for i in range(0,len(self.Encoder_FC),2): 
            x = self.Encoder_FC[i](x) # Linear layer
            x = self.Encoder_FC[i+1](x) # Batchnorm
            x = relu(x)
            if self.use_dropout:
                x = dropout(x, p=self.do_p_lin)
        return (x)

class Aux_encoder(nn.Module):
    def __init__(self, params):
        super(Aux_encoder, self).__init__()
        self.params = params
        # Auxillary encoder
        Encoder_FC = nn.ModuleList()
        in_weights = params['lin_layer'][-1]
        for i in range(params['num_aux']):
            Encoder_FC.append(Linear(in_features=in_weights, out_features=params['aux_layer'][i]))
            Encoder_FC.append(BatchNorm1d(params['aux_layer'][i]))
            in_weights = params['aux_layer'][i]
        Encoder_FC.append(Linear(in_features=params['aux_layer'][-1], out_features=params['aux_variables']*2))
        Encoder_FC.append(BatchNorm1d(params['aux_variables']*2))
        self.add_module("Encoder_aux", Encoder_FC)
            
    def forward(self,a):
        '''
        Inputs:
        a - output from encoder [batch_size,num_features]
        Outputs:
        q_a_mu - mean of encoder [batch_size, aux_variables]
        q_a_log_var - log variance of aux. encoder [batch_size, aux_variables]
        '''
        for i in range(0,len(self.Encoder_aux),2):
            a = self.Encoder_aux[i](a)
            a = self.Encoder_aux[i+1](a)
            a = relu(a)
            if self.params['use_dropout']:
                a = dropout(a, p=self.params['do_p_lin'])
        q_a_mu, q_a_log_var = torch.chunk(a, 2, dim=-1) # divide to mu and sigma
        return q_a_mu, q_a_log_var
 

'''
Decoders
'''
class Decoder(nn.Module):
    def __init__(self, params, layer_size):
        super(Decoder, self).__init__()
        self.layer_size = layer_size
        
        self.final_dim = compute_final_dimension(height=params['img_dimension'][0], 
                            width=params['img_dimension'][1], last_num_channels=params['conv_out_channels'][-1], 
                            conv_kernel=params['conv_kernel'], conv_padding=params['conv_padding'], 
                            conv_stride=params['conv_stride'], pool_kernel=params['pool_kernel'], 
                            pool_padding=params['pool_padding'], pool_stride=params['pool_stride'], num_conv=params['num_conv'])

        # Initialize fully connected layers from latent space to convolutional layers
        Decoder_FC = nn.ModuleList()
        Decoder_FC.append(Linear(in_features=params['latent_features']+params['num_classes'], out_features=params['lin_layer'][-1]))
        Decoder_FC.append(BatchNorm1d(params['lin_layer'][-1]))
        for i in reversed(range(params['num_lin'])):
            if i == 0:
                out_weights = self.final_dim[0]
            else:
                out_weights = params['lin_layer'][i-1]
            Decoder_FC.append(Linear(in_features=params['lin_layer'][i], out_features=out_weights))
            Decoder_FC.append(BatchNorm1d(out_weights))
        self.add_module("Decoder_FC",Decoder_FC)
        
        # Convolutional layers of the decoder
        Decoder_conv = nn.ModuleList()
        for i in reversed(range(params['num_conv'])):
            if i == 0:
                output_channels = params['input_channels']*2
            else:
                output_channels = params['conv_out_channels'][i-1] 
                
            Decoder_conv.append(ConvTranspose2d(in_channels=params['conv_out_channels'][i],
                                                out_channels=output_channels,
                                                kernel_size=params['conv_kernel'][i],
                                                stride=params['conv_stride'][i],
                                                padding=params['conv_padding'][i]))
            Decoder_conv.append(BatchNorm2d(output_channels))
        self.add_module("Decoder_conv",Decoder_conv)
    
    def forward(self,z,y):
        '''
        Inputs:
        z - variables from latent space [batch_size, num_samples, latent_features]
        y - onehot encoded label [batch_size, num_samples, num_classes]
        Outputs:
        x - reconstruction [batch_size, num_samples , channels*2, height, width]
        '''
        x = torch.cat([z,y],dim=-1)
        # Fully connected layers of decoder
        for i in range(0,len(self.Decoder_FC),2):
            x = self.Decoder_FC[i](x)
            x = x.permute(0,2,1)
            x = self.Decoder_FC[i+1](x)
            x = x.permute(0,2,1)
            x = relu(x)
            x = dropout(x, p=self.params['do_p_lin'])
        x = x.reshape(-1, self.Decoder_conv[0].in_channels, self.final_dim[1], self.final_dim[2])
        
        # Convolutional layers of decoder
        curr_layer = len(self.Decoder_conv)//2-1
        for i in range(0,len(self.Decoder_conv),2):
            x = interpolate(x,size = [self.layer_size[curr_layer],self.layer_size[curr_layer]],
                                      mode = 'bilinear', 
                                      align_corners = False)
            curr_layer -=1
            x = self.Decoder_conv[i](x) # Convolutional layers
            x = self.Decoder_conv[i+1](x) # BatchNorm
            x = relu(x)
            if self.params['use_dropout']:
                x = dropout2d(x, p=self.params['do_p_conv'])
        return x.view(self.params['batch_size'],-1,self.params['channels']*2,self.params['img_dimension'][0],self.params['img_dimension'][1])

class Aux_decoder(nn.Module):
    def __init__(self, params):
        super(Aux_decoder, self).__init__()
        self.params = params
        Decoder_aux = nn.ModuleList()
        for i in range(params['num_aux_decoder']):
            if i == 0:
                in_weights = params['latent_features']+ params['lin_layer'][-1] + params['num_classes']
            else:
                in_weights = params['aux_decoder_layers'][i-1]
            Decoder_aux.append(Linear(in_features=in_weights, out_features=params['aux_decoder_layers'][i]))
            Decoder_aux.append(BatchNorm1d(params['aux_decoder_layers'][i]))
        Decoder_aux.append(Linear(in_features=params['aux_decoder_layers'][-1], out_features=params['aux_variables']*2))
        Decoder_aux.append(BatchNorm1d(params['aux_variables']*2))
        self.add_module("Decoder_aux", Decoder_aux)
        
    def forward(self, xz, y):
        xzy = torch.cat([xz,y],dim=-1)
        for i in range(0,len(self.Decoder_aux),2):
            xzy = self.Decoder_aux[i](xzy)
            xzy = xzy.permute(0,2,1) # change dimensions so it works with batchnorm
            xzy = self.Decoder_aux[i+1](xzy)
            xzy = xzy.permute(0,2,1)
            xzy = relu(xzy)
            if self.params['use_dropout']:
                xzy = dropout(xzy, p=self.params['do_p_lin'])
        a = xzy
        return a        

'''
Classifier
'''
class Classifier(nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()
        self.params = params
        Classifier_layers = nn.ModuleList()
        if params['aux_variables']> 0:
            in_weights = params['lin_layer'][-1]+params['aux_variables']
        else:
            in_weights = params['lin_layer'][-1]
        for i in range(params['num_classes']):
            Classifier_layers.append(Linear(in_features=in_weights, out_features=params['classifier_layer'][i]))
            Classifier_layers.append(BatchNorm1d(params['classifier_layer'][i]))
            in_weights = params['classifier_layer'][i]
        Classifier_layers.append(Linear(in_features=params['classifier_layer'][-1], 
                                        out_features = params['num_classes']))
        self.add_module("Classifier", Classifier_layers)
    
    def forward(self,xa):
        '''
        Inputs:
        xa - output from encoder and samples from aux. encoder.[bacth_size, num_samples, num_features]
        Outputs:
        y - class prediction [batch_size,num_samples,num_classes]
        '''
        for i in range(0,len(self.Classifier),2):
            xa = self.Classifier[i](xa)
            if i < len(self.Classifier)-1:
                xa = xa.permute(0,2,1) # If samples exist batchnorm is expecting in different form as else. so permute.
                xa = self.Classifier[i+1](xa)
                xa = xa.permute(0,2,1) # and permute back again.
            xa = relu(xa)
            if self.params['use_dropout']:
                xa = dropout(xa, p=0.3)
        return softmax(xa,dim=-1)

   
'''
Helper functions for constructing the network
'''
# Calculating the dimensions 
def compute_conv_dim(height, width, kernel_size, padding_size, stride_size):
    height_new = int((height - kernel_size + 2 * padding_size) / stride_size + 1)
    width_new =  int((width  - kernel_size + 2 * padding_size) / stride_size + 1)
    return [height_new, width_new]

def compute_final_dimension(height=None, width=None, last_num_channels=None, num_conv=None, conv_kernel=None, conv_padding=None, 
                            conv_stride=None, pool_kernel=None, pool_padding=None, pool_stride=None):
    # First conv layer
    CNN_height = height
    CNN_width = width
    
    for i in range(num_conv):
        # conv layer
        CNN_height, CNN_width = compute_conv_dim(CNN_height, CNN_width, conv_kernel[i], conv_padding[i], conv_stride[i])
        # maxpool layer
        CNN_height, CNN_width = compute_conv_dim(CNN_height, CNN_width, pool_kernel, pool_padding, pool_stride)
    final_dim = CNN_height * CNN_width * last_num_channels
    return [final_dim, CNN_height, CNN_width]

def normalize(x):
    tmp = x-torch.min( torch.min(x,dim = 2, keepdim = True)[0] ,dim = 3, keepdim = True)[0]
    if torch.sum(torch.isnan(tmp))>0:
        print("nan of tmp",torch.sum(torch.isnan(tmp)))
    return tmp/(torch.max( torch.max(tmp,dim = 2, keepdim = True)[0] ,dim = 3, keepdim = True)[0] + 1e-8)  

def gaussian_sample(mu,log_var, num_samples, latent_features):    
    # Don't propagate gradients through randomness
    with torch.no_grad():
        batch_size = mu.size(0)
        epsilon = torch.randn(batch_size, num_samples, latent_features)
            
    if cuda:
        epsilon = epsilon.cuda()
        
    sigma = torch.exp(log_var/2)
        
    # We will need to unsqueeze to turn
    # (batch_size, latent_dim) -> (batch_size, 1, latent_dim)
    if len(mu.shape) == 2:
        z = mu.unsqueeze(1) + epsilon * sigma.unsqueeze(1)
    else:
        z = mu.repeat(1,num_samples//mu.size(1),1) + epsilon * sigma.repeat(1,num_samples//sigma.size(1),1)
    return z

def output_recon(x):
    # Shape of x_mean: [batch_size, num_samples, channel, height, width]
    x_mean, x_log_var = torch.chunk(x, 2, dim=2) # the mean and log_var reconstructions from the decoder

    # extract one sample to show as image. (if we take the mean, it becomes blurry.) 
    x_hat = x_mean[:,1,].unsqueeze(1) # used for the reconstruction image.
    x_log_var = softplus(x_log_var)
    
    # Mean over samples
    #x_log_var= torch.mean(x_log_var, dim=1) # used for the loss
    #x_mean = torch.mean(x_mean,dim=1) # used for the loss

    return x_hat, x_log_var, x_mean

def get_layer_sizes(img_dimension, conv_out_channels, conv_kernel, conv_padding, conv_stride,
                    pool_kernel, pool_padding, pool_stride):
    img_height, img_width = img_dimension
    num_layers = len(conv_out_channels)
    tmp_height = img_height
    tmp_width = img_width
    # initiate variable
    layer_sizes = []
    # Run through layers. 
    for i in range(num_layers):
        tmp_height, tmp_width = compute_conv_dim(tmp_height, tmp_width, conv_kernel[i], conv_padding[i], conv_stride[i])
        layer_sizes.append(tmp_height)
        tmp_height, tmp_width = compute_conv_dim(tmp_height, tmp_width, pool_kernel, pool_padding, pool_stride)   
    return layer_sizes

def marginalizeY(x,num_classes,dimension):
    # Chunk y samples and put in dim 0 to be able to sum over y
    x = torch.stack(torch.chunk(x, num_classes, dim=dimension))
    # Take mean of y-sampling
    return(torch.mean(x,dim = 0))

