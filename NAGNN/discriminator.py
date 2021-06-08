#encoding=utf-8
'''
'''
import numpy as np
import tensorflow as tf

def discriminatorModel(options, variablesMap, ffd_drop, lbl_in, lbl_in_1, adj_0, mask_0, adj_1, mask_1, ftr_in, isSupervised_flag, isReal_flag, isOnlyAddZerosCol_flag, isDLoss_flag, isPreTrain_flag):
    """
    discriminator
    """
    logits = GCNModel(ftr_in, adj_0, mask_0, adj_1, mask_1, variablesMap, ffd_drop, activation=options['activation'])
    
    loss, lossL2, accuracy, predLabels, trueLabels = discriminatorLoss(options, variablesMap, logits, lbl_in, lbl_in_1, isSupervised_flag, isReal_flag, isOnlyAddZerosCol_flag, isDLoss_flag, isPreTrain_flag)
    
    return loss, lossL2, accuracy, predLabels, trueLabels
    
    
def discriminatorLoss(options, variablesMap, logits, lbl_in, lbl_in_1, isSupervised_flag, isReal_flag, isOnlyAddZerosCol_flag, isDLoss_flag, isPreTrain_flag):    
    """
    discriminator loss
    """
    log_resh = tf.reshape(logits, [-1, options['all_class_num']]) 
    lab_resh = tf.reshape(lbl_in, [-1, options['all_class_num']]) 
    lab_resh_1 = tf.reshape(lbl_in_1, [-1, options['all_class_num']]) 
    msk_in = tf.ones(shape=tf.shape(lab_resh)[0], dtype=tf.bool)
    msk_resh = tf.reshape(msk_in, [-1]) 
    
    loss_un = masked_softmax_cross_entropy_unsupervised(log_resh, msk_resh, isReal_flag)
    loss_su = masked_softmax_cross_entropy_supervised(log_resh, lab_resh, msk_resh)
    loss_su_1 = masked_softmax_cross_entropy_supervised(log_resh, lab_resh_1, msk_resh)
    
    loss = (1.0 - isSupervised_flag) * loss_un + isSupervised_flag * ( isOnlyAddZerosCol_flag * loss_su + (1.0-isOnlyAddZerosCol_flag) * loss_su_1 )
    
    vars = tf.trainable_variables()
    lossL2 = loss + (isDLoss_flag * (isPreTrain_flag * (tf.nn.l2_loss(variablesMap['D_W0']) + tf.nn.l2_loss(variablesMap['D_W1']) + tf.nn.l2_loss(variablesMap['D_b0']) + tf.nn.l2_loss(variablesMap['D_b1'])) * options['pre_l2_coef'] + (1.0-isPreTrain_flag) * (tf.nn.l2_loss(variablesMap['D_W0']) + tf.nn.l2_loss(variablesMap['D_W1'])) * options['GAN_l2']))  + ((1.0-isDLoss_flag) * (tf.nn.l2_loss(variablesMap['G_MLP_W']) + tf.nn.l2_loss(variablesMap['G_MLP_b']))) * options['GAN_l2']
    
    accuracy = masked_accuracy(log_resh, lab_resh, msk_resh)
    predLabels, trueLabels = predictLabels(log_resh, lab_resh)
    
    return loss, lossL2, accuracy, predLabels, trueLabels


def GCNModel(ftr_in, adj_0, mask_0, adj_1, mask_1, variablesMap, 
              ffd_drop, activation=tf.nn.elu):
    """
    GCN model
    """
    h_1 = layerModel(ftr_in, adj_1, mask_1, variablesMap['D_W0'], variablesMap['D_b0'], ffd_drop, activation)
    h_2 = layerModel(h_1, adj_0, mask_0, variablesMap['D_W1'], variablesMap['D_b1'], ffd_drop, activation)
    logits = h_2
    return logits 

def layerModel(seq, adj, mask, W, b, in_drop=0.0, activation=tf.nn.elu):
    """
    GCN layer
    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.matmul(seq, W) 
        
        seq_fts = tf.nn.embedding_lookup(seq_fts, adj) 
        seq_fts = seq_fts * mask[:,:,None]
        ret = tf.reduce_sum(seq_fts, axis=1) + b
        
        if in_drop != 0.0: 
            ret = tf.nn.dropout(ret, 1.0 - in_drop)
        return activation(ret)  


def masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy_unsupervised(logits, mask, isReal_flag):
    """
    softmax cross entropy for unsupervised
    """
    logits = tf.nn.softmax(logits)
    off = 1e-6 
    loss_r = -tf.log(tf.reduce_sum(logits[:, :-1], axis=1) + off)  
    loss_f = -tf.log(logits[:, -1] + off)  
    
    loss = isReal_flag * loss_r + (1.0 - isReal_flag) * loss_f
    
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss) 


def masked_softmax_cross_entropy_supervised(logits, labels, mask):
    """
    softmax cross entropy for unsupervised
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=(tf.cast(labels, tf.float32)))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(logits, labels, mask):
    """Accuracy with masking.
    """
    logits = logits[:,:-1] 
    labels = labels[:,:-1] 
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def predictLabels(logits, trueLabels):
    """prediction
    """
    logits = logits[:,:-1] 
    trueLabels = trueLabels[:,:-1] 
    predicted = tf.nn.sigmoid(logits) 
    pred_labels = tf.argmax(predicted, axis=1)
    true_labels = tf.argmax(trueLabels, axis=1)
    return pred_labels, true_labels

def training(loss, lr, l2_coef):
    # weight decay
    # Returns all variables created with `trainable=True`.
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                       in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
    # optimizer
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    # training op
    train_op = opt.minimize(loss+lossL2)
    return train_op
    
    
    
    