#encoding=utf-8
'''

'''
import numpy as np
import tensorflow as tf


def generatorFeatureByAdjModel(options, variablesMap, idealFeatures, features_array, feature_self_mask, feature_cosp_mask, sigma, G_pretrain_weight):
    """
    generator
    """
    samplings = tf.random_normal(shape=tf.shape(idealFeatures), mean=0.0, stddev=sigma)
    fakeFeature = G_pretrain_weight * samplings + idealFeatures
    fakeFeature = tf.nn.elu(tf.matmul(fakeFeature, variablesMap['G_MLP_W']) + variablesMap['G_MLP_b']) 
    fakeFeature_nor = fakeFeature / tf.reduce_sum(fakeFeature, axis=1)[:,None] 
    
    fakeFeaturesArray = features_array * (1.0-feature_self_mask)[:,None] + tf.matmul(feature_cosp_mask, fakeFeature_nor) 
    
    multiple = 1000 
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(idealFeatures*multiple - fakeFeature*multiple), axis=1))
    
    lossL2 = loss + (tf.nn.l2_loss(variablesMap['G_MLP_W']) + tf.nn.l2_loss(variablesMap['G_MLP_b'])) * options['GAN_l2']
    
    return fakeFeaturesArray, loss, lossL2
    
    
    
    