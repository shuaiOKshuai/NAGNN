#encoding=utf-8
'''
'''
import numpy as np
import os 
import configparser
import AdGCNTraining


cf = configparser.SafeConfigParser()
cf.read("paramsConfigPython")

root_dir = cf.get("param", "root_dir") 
dataset = cf.get("param", "dataset") # dataset name
root_dir = root_dir + dataset + '/'
gpu = cf.get("param", "gpu") # gpu id

os.environ["CUDA_VISIBLE_DEVICES"] = gpu 


hid_units = [int(i) for i in cf.get("param", "hid_units").split(',')] 
sigma = cf.getfloat("param", "sigma") #the default std for normal distribution for sampling
lr_D = cf.getfloat("param", "lr_D") # learning rate
lr_G = cf.getfloat("param", "lr_G") # learning rate
alpha = cf.getfloat("param", "alpha") # the weight for fake data
m_D = cf.getint("param", "m_D") # the time for fake nodes wrt one real node
pre_l2_coef = cf.getfloat("param", "pre_l2_coef") # coefficient of l2 regularization 
GAN_D_l2_coef = cf.getfloat("param", "GAN_D_l2_coef") # coefficient of l2 regularization 
GAN_G_l2_coef = cf.getfloat("param", "GAN_G_l2_coef") # coefficient of l2 regularization 
l2_coef_param = cf.getfloat("param", "l2_coef_param") # coefficient of l2 regularization 
dropout = cf.getfloat("param", "dropout") # dropout
D_pretrain_epoch = cf.getint("param", "D_pretrain_epoch") # pretrain epoch for D
D_pretrain_patience = cf.getint("param", "D_pretrain_patience") # pretrain patience epoch for D
G_pretrain_epoch = cf.getint("param", "G_pretrain_epoch") # pretrain epoch for G
G_pretrain_param = cf.getfloat("param", "G_pretrain_param") # G_pretrain_param
G_pretrain_min_loss  = cf.getfloat("param", "G_pretrain_min_loss") # G_pretrain_param
epoch_num = cf.getint("param", "epoch_num") # epoch
inner_epoch_D = cf.getint("param", "inner_epoch_D") # in each batch data, the inner training epochs for D
inner_epoch_G = cf.getint("param", "inner_epoch_G") # in each batch data, the inner training epochs for G
patience = cf.getint("param", "patience") # patience for training 


# train NAGCN
AdGCNTraining.adGCNTraining(
    root_dir, 
    dataset, 
    hid_units, 
    sigma,
    lr_D,
    lr_G, 
    alpha,
    m_D,
    pre_l2_coef, 
    GAN_D_l2_coef,
    GAN_G_l2_coef,
    l2_coef_param,
    dropout,
    D_pretrain_epoch, 
    D_pretrain_patience,
    G_pretrain_epoch, 
    G_pretrain_param,
    G_pretrain_min_loss,
    epoch_num, 
    inner_epoch_D,
    inner_epoch_G, 
    patience)

