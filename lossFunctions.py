# -*- coding: utf-8 -*-

import math
import torch

def Gaussian_density(sample_img,mu_img,log_var_img):
    """
    Log likelihood of a Gaussian density (http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html):
    sum_1^n [ -0.5 * log(2π) - log(var)/2 - (x - mean)^2 / (2*var + eps) ]

    All inputs are w/ dimensions: ? [batch_size, channels, width, height]

    :param sample_img: sample x
    :param mu_img: mean
    :param log_var_img: logarithm of the variance
    :return: log likelihood averaged across channels and pixels, output dim [batch_size]
    """
    c = - 0.5 * math.log(2 * math.pi)
    log_likelihood = c - log_var_img/2 - (sample_img - mu_img)**2/(2 * torch.exp(log_var_img + 1e-9))
    log_likelihood = torch.mean(log_likelihood, dim=(1, 2, 3))
    return log_likelihood


def kl_a_calc(q_a, q_mu, q_log_var,p_mu, p_log_var):
    """
    Kullback-Liebler divergence for auxiliary variable

    :param q_a: [batch_size, no_samples, latent_features]
    :param q_mu: [batch_size, latent_features]
    :param q_log_var: [batch_size, latent_features]
    :param p_mu: [batch_size, no_samples, latent_features]
    :param p_log_var: [batch_size, no_samples, latent_features]
    :return:
    """
    
    def log_gaussian(x, mu, log_var):
        log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
        log_pdf = torch.sum(log_pdf, dim=1)  # sum over each over the observations (mu + log_var*epsilon)
        log_pdf = torch.sum(log_pdf, dim=1)  # sum over q_a, i.e. latent features
        return log_pdf

    # put in middle dimension, i.e. no_samples
    q_mu      = q_mu.unsqueeze(1)
    q_log_var = q_log_var.unsqueeze(1)
    p_mu      = p_mu.unsqueeze_(1)
    p_log_var = p_log_var.unsqueeze_(1)
    # densities of each disitribution 
    qz = log_gaussian(q_a,q_mu,q_log_var)
    pz = log_gaussian(q_a,p_mu,p_log_var)
    # kl divergence
    kl = qz - pz
    
    return kl
        
def ELBO_loss(params, sample_img, outputs, y = None, kl_warmup = None):
    # Parameter in deterministic warmup for KL divergence
    beta = 1 if kl_warmup is None else kl_warmup
    
    # Weighting kl's and likelihood
    w1 = 0.5
    w2 = 0.5
    
    if y == None: 
        ELBO = []
        # KL divergence when assuming q(z) being standard normal distributed...
        
        kl_x = -0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu']**2 - torch.exp(outputs['log_var']), dim=1)
        kl_x = torch.sum(kl_x,dim=1) # sum over the features
        
        # divide into samples of y.
        x_mean = torch.chunk(outputs['x_mean'], params["num_classes"], dim=1)
        x_log_var = torch.chunk(outputs['x_log_var'], params["num_classes"],dim=1)
        a_mean = torch.chunk(outputs['p_a_mu'], params["num_classes"],dim=1)
        a_log_var = torch.chunk(outputs['p_a_log_var'], params["num_classes"],dim=1)
        for j in range(params["num_classes"]):
            likelihood = Gaussian_density(sample_img, torch.mean(x_mean[j],dim = 1), torch.mean(x_log_var[j],dim = 1))
            if params["aux_variables"] > 0:
                kl_a = kl_a_calc(outputs["q_a"], outputs["q_a_mu"], outputs["q_a_log_var"],
                                 torch.mean(a_mean[j], dim = 1), torch.mean(a_log_var[j], dim = 1))
                kl = w1 * kl_x + (1 - w1) * kl_a
            else:
                kl_a = torch.Tensor([0])
                kl = kl_x
            ELBO.append(w2 * likelihood - (1 - w2) * beta * kl)
        L = torch.stack(ELBO).t()
        #L = torch.cat( (torch.unsqueeze(ELBO[0],1),torch.unsqueeze(ELBO[1],1)),dim =1 )
        # Calculate entropy H(q(y|x)) and sum over all labels
        logits = torch.mean(outputs['y_hat'],dim = 1) 
        # Minizing the negative entropy is equivalent to maximizing the certainty
        # of the decisions (logits goes toward 0 or 1 (in a two class problem))
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1) 
        L = torch.sum(torch.mul(logits, L), dim=-1)
        
        # Equivalent to -U(x)
        U = L - H
        
        return -torch.mean(U), -torch.mean(H), -torch.mean(L),\
               (1 - w2) * beta * torch.mean(kl), -w2 * torch.mean(likelihood),\
               w1*torch.mean(kl_x), (1-w1)*torch.mean(kl_a)
    else:
        likelihood = Gaussian_density(sample_img,
                                      torch.mean(outputs['x_mean'], dim=1),
                                      torch.mean(outputs['x_log_var'], dim=1))
        kl_x = -0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu']**2 - torch.exp(outputs['log_var']), dim=1)
        kl_x = torch.sum(kl_x,dim=1) # sum over the features
        if params["aux_variables"] > 0:
            kl_a = kl_a_calc(outputs["q_a"], outputs["q_a_mu"], outputs["q_a_log_var"],
                             torch.mean(outputs["p_a_mu"], dim=1), torch.mean(outputs["p_a_log_var"], dim=1))
            kl = w1 * torch.mean(kl_x) + (1 - w1) * torch.mean(kl_a)
        else:
            kl_a = torch.Tensor([0])
            kl = torch.mean(kl_x)
        ELBO = w2 * torch.mean(likelihood) - (1 - w2) * beta * kl    
        # Notice minus sign as we want to maximise ELBO
        return -ELBO, (1 - w2) * beta * kl, -w2 * torch.mean(likelihood), w1*torch.mean(kl_x), (1-w1)*torch.mean(kl_a)
    
    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    # In the case of the KL-divergence between diagonal covariance Gaussian and 
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch

# Function to normalize a single image
def normalize(x):
    # Input: [1, height, width]
    x_shape = x.shape
    x = x.view(1,-1)
    x = x - torch.min(x)
    x = x / (torch.max(x) + 1e-8)
    if torch.sum(torch.isnan(x))>0:
        print("nan of tmp",torch.sum(torch.isnan(x)))
    return x.view(x_shape)

# Function to calculate balanced accuracy
def balanced_accuracy(logits, y_hot, Do_print=None):
    # Input:
    # logits: [batch_size, num_samples, num_classes]
    # y_hot: [batch_size, num_samples, num_classes]
    
    logits = logits.cpu().detach()
    y_hot = y_hot.cpu().detach()
    
    logits = logits.view(-1,2)
    y_hot = y_hot.view(-1,2)
    
    TP = torch.sum(y_hot[:,0]*torch.round(logits[:,0]))   # True posistive
    FP = torch.sum(y_hot[:,1]*torch.round(logits[:,0]))   # False positive
    FN = torch.sum(y_hot[:,0]*torch.round(logits[:,1]))   # False negative
    TN = torch.sum(y_hot[:,1]*torch.round(logits[:,1]))   # True negative

    P = TP + FN
    N = FP + TN
    
    if Do_print != None:
        print("TP: ",TP)
        print("FP: ",FP)
        print("TN: ",TN)
        print("FN: ",FN)
        print("P: ",P)
        print("N: ",N)

    acc = torch.sum(TP/P + TN/N)/2
        
    return acc

# Function to calculate balanced binary crossentropy
def balanced_binary_cross_entropy( logits, y_hot):
    # Input:
    # logits: [batch_size, num_samples, num_classes]
    # y_hot: [batch_size, num_samples, num_classes]
    
    classWeight = torch.FloatTensor([torch.sum(y_hot[:,1,0])/torch.sum(y_hot[:,1,]), torch.sum(y_hot[:,1,1])/torch.sum(y_hot[:,1,])]).cuda(device=0)
    class_loss_0 = 0
    class_loss_1 = 0

    batch_size = logits.shape[0]
    
    for i in range(0, batch_size):
        tmp = torch.mean(y_hot, dim = 1)[i][0] 
        if tmp == 0:
            class_loss_0 += torch.nn.functional.binary_cross_entropy(logits[i], y_hot[i])
        else:
            class_loss_1 += torch.nn.functional.binary_cross_entropy(logits[i], y_hot[i])
        
    #bal_binary_cross_entropy = classWeight[0]*class_loss_0 + classWeight[1]*class_loss_1
    bal_binary_cross_entropy = (0.5/classWeight[0])*class_loss_0 + (0.5/classWeight[1])*class_loss_1
        
    return bal_binary_cross_entropy/batch_size

def balanced_accuracy_test( logits, y_hot, Do_print=None):
    # Input:
    # logits: [batch_size, num_samples, num_classes]
    # y_hot: [batch_size, num_samples, num_classes]
    
    logits = logits.cpu().detach()
    y_hot = y_hot.cpu().detach()
    
    logits = logits.view(-1,2)
    y_hot = y_hot.view(-1,2)
    
    TP = torch.sum(y_hot[:,0]*torch.round(logits[:,0]))   # True posistive
    FP = torch.sum(y_hot[:,1]*torch.round(logits[:,0]))   # False positive
    FN = torch.sum(y_hot[:,0]*torch.round(logits[:,1]))   # False negative
    TN = torch.sum(y_hot[:,1]*torch.round(logits[:,1]))   # True negative

    P = TP + FN
    N = FP + TN
    
    if Do_print != None:
        print("TP: ",TP)
        print("FP: ",FP)
        print("TN: ",TN)
        print("FN: ",FN)
        print("P: ",P)
        print("N: ",N)

    acc = torch.sum(TP/P + TN/N)/2
        
    return acc, TP, FP, FN, TN, P, N


    


