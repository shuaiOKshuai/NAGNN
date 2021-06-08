#encoding=utf-8
'''
tool functions
'''
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pickle as pkl
import networkx as nx
import sys
import scipy.sparse as sp
import time
from numpy import dtype
import random
from collections import Counter

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features_new = r_mat_inv.dot(features)
    features_01 = features.todense()
    return features_new.todense(), sparse_to_tuple(features_new), features_01


def load_data_dblp(nodeFile, edgeFile, trainFile, valFile, testFile):
    """
    load other data
    """
    print(nodeFile)
    print(edgeFile)
    node_num = 0
    feature_num = 0
    features = None
    index = 0
    nodeLabels=[]
    labelsSet = set()
    lineCount = 0
    with open(nodeFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                if len(arr)==2 and lineCount==0:
                    print(arr)
                    node_num=int(arr[0])
                    feature_num=int(arr[1])
                    print('node num == ', node_num)
                    print('feature num == ', feature_num)
                    features=np.zeros((node_num, feature_num))
                    lineCount += 1
                    continue
                features[index]=arr[2:]
                nodeLabels.append(int(arr[1])) 
                labelsSet.add(int(arr[1]))
                index+=1
                lineCount += 1
    label_num = len(labelsSet)
    adj=np.zeros((node_num,node_num))
    with open(edgeFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                adj[int(arr[0]), int(arr[1])]=1.0
    y_train = np.zeros((node_num,label_num))
    y_val = np.zeros((node_num,label_num))
    y_test = np.zeros((node_num,label_num))
    train_mask = np.zeros((node_num,))
    val_mask = np.zeros((node_num,))
    test_mask = np.zeros((node_num,))
    
    with open(trainFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                id=int(tmp)
                y_train[id, nodeLabels[id]] = 1.0
                train_mask[id] = 1.0
    with open(valFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                id=int(tmp)
                y_val[id, nodeLabels[id]] = 1.0
                val_mask[id] = 1.0
    with open(testFile) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                id=int(tmp)
                y_test[id, nodeLabels[id]] = 1.0
                test_mask[id] = 1.0
    
    features_01 = features
    features = features / features.sum(axis=1)[:,None]
    
    features = np.mat(features)
    features_01 = np.mat(features_01)
    
    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)
    test_mask = np.array(test_mask, dtype=np.bool)
    
    return adj, features, features_01, y_train, y_val, y_test, train_mask, val_mask, test_mask

def micro_macro_f1_removeMiLabels(y_true, y_pred, labelsList):
    return f1_score(y_true, y_pred, labels = labelsList, average="micro"), f1_score(y_true, y_pred, average="macro")

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_unlabeled(rootdir, dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """
    load data for unlabeled data
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(rootdir+"ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0): 
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(rootdir+"ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil() 
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    node_num = adj.shape[0]
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_unlabeled = range(len(y)+500, node_num-len(idx_test)) 

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    unlabeled_mask = sample_mask(idx_unlabeled, labels.shape[0])
    
    
    y_train = np.zeros(labels.shape) 
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, unlabeled_mask, test_mask, idx_unlabeled


def preprocess_adj_normalization(adj):
    """
    normalization
    """
    num_nodes = adj.shape[0] 
    adj = adj + np.eye(num_nodes)  
    adj[adj > 0.0] = 1.0
    D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5)) 
    adjNor = np.dot(np.dot(D_, adj), D_)
    return adj, adjNor

def get_minibatches_idx(startID, endID, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(start=startID, stop=endID, dtype=np.int32)
    n=endID-startID 

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def processNodeInfo(adj, mask_nor, node_num):
    """
    process node info
    """
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)
    neis = np.zeros((node_num,max_nei_num), dtype=np.int32)
    neis_mask = np.zeros((node_num,max_nei_num), dtype=np.float32)
    neis_mask_nor = np.zeros((node_num,max_nei_num), dtype=np.float32)
    neighboursDict = []
    inner_index = 0 
    for i in range(node_num):
        inner_index = 0
        nd = [] 
        for j in range(node_num):
            if adj[i][j]==1.0: 
                neis[i][inner_index] = j 
                neis_mask[i][inner_index] = 1.0
                neis_mask_nor[i][inner_index] = mask_nor[i][j]
                if i!=j: 
                    nd.append(j)
                inner_index += 1
        neighboursDict.append(nd)
    
    return neis, neis_mask, neis_mask_nor, neighboursDict


def generateDataByGivenNodes_su_featuresProcess(adj, adjNor, mask, features, features_01, node_num, neisMatrix, neisMatrix_mask, neisMatrix_mask_nor, lables=None):
    """
    prepare data given nodes for supervised setting, considering features
    """
    features = features.A
    features_01 = features_01.A
    wholeWidth = neisMatrix.shape[1] 
    ids = [i for i in range(node_num) if mask[i]==True]
    valid_node_num = len(ids)
    adj_0 = neisMatrix[ids] 
    mask_0 = neisMatrix_mask[ids] 
    mask_0_nor = neisMatrix_mask_nor[ids] 
    
    adj_0_max_nei_num = np.max(np.sum(mask_0, axis=1)).astype(np.int32)
    adj_0 = adj_0[:,:adj_0_max_nei_num]
    mask_0 = mask_0[:,:adj_0_max_nei_num]
    mask_0_nor = mask_0_nor[:,:adj_0_max_nei_num]
    
    valid_node_num_adj_0 = np.sum(mask_0).astype(np.int32)
    adj_0_new = np.zeros_like(adj_0).astype(np.int32) 
    adj_1 = np.zeros((valid_node_num_adj_0, wholeWidth)).astype(np.int32)
    mask_1 = np.zeros((valid_node_num_adj_0, wholeWidth))
    mask_1_nor = np.zeros((valid_node_num_adj_0, wholeWidth))
    mask_1_self = np.zeros((valid_node_num_adj_0, wholeWidth)) 
    index = 0 
    adj_1_map = np.zeros((valid_node_num_adj_0,)).astype(np.int32) 
    for i in range(adj_0.shape[0]):
        nodeId = ids[i]
        valid_num_row = np.sum(mask_0[i]).astype(np.int32)
        valid_ids_row = adj_0[i, :valid_num_row]
        adj_1[index:index+valid_num_row] = neisMatrix[valid_ids_row] 
        mask_1[index:index+valid_num_row] = neisMatrix_mask[valid_ids_row]
        mask_1_nor[index:index+valid_num_row] = neisMatrix_mask_nor[valid_ids_row]
        mask_1_self[index:index+valid_num_row] = np.where(adj_1[index:index+valid_num_row]==nodeId, 1.0, 0.0) 
        mask_1_self[index:index+valid_num_row] = (mask_1_self[index:index+valid_num_row] * mask_1[index:index+valid_num_row]).astype(np.int32) 
        adj_0_new[i,:valid_num_row] = np.arange(index,index+valid_num_row)
        adj_1_map[index:index+valid_num_row]=i 
        
        index += valid_num_row
    adj_1_max_nei_num = np.max(np.sum(mask_1, axis=1)).astype(np.int32)
    adj_1 = adj_1[:,:adj_1_max_nei_num]
    mask_1 = mask_1[:,:adj_1_max_nei_num]
    mask_1_nor = mask_1_nor[:,:adj_1_max_nei_num]
    mask_1_self = mask_1_self[:,:adj_1_max_nei_num]
    
    lbl = lables[mask] 
    extra = np.zeros((valid_node_num,1), dtype=np.float32)
    lbl = np.concatenate((lbl, extra), axis=1)
    lbl_1 = np.zeros_like(lbl, dtype=np.float32)
    lbl_1[:,-1] = 1.0 
    
    idealFeatures = np.dot(adjNor[ids], features) 
    idealFeatures_01 = np.dot(adjNor[ids], features_01) 
    
    index = 0 
    allFeaturesNum = np.sum(mask_1).astype(np.int32)
    adj_1_new = np.zeros_like(adj_1, dtype=np.int32)
    feature_num = features.shape[1] 
    allFeatures = np.zeros((allFeaturesNum,feature_num))
    feature_self_mask = np.zeros((allFeaturesNum,))
    feature_cosp_mask = np.zeros((allFeaturesNum,valid_node_num))
    for i in range(mask_1.shape[0]):
        validNum_row = np.sum(mask_1[i]).astype(np.int32)
        valid_ids_row = adj_1[i,:validNum_row] 
        allFeatures[index:index+validNum_row] = features[valid_ids_row]
        adj_1_new[i,:validNum_row] = np.arange(index,index+validNum_row)
        feature_self_mask[index:index+validNum_row] = mask_1_self[i,:validNum_row]
        feature_cosp_mask[index:index+validNum_row,adj_1_map[i]] = mask_1_self[i,:validNum_row]
        index += validNum_row
    
    return idealFeatures, idealFeatures_01, lbl, lbl_1, adj_0_new, mask_0, mask_0_nor, adj_1_new, mask_1, mask_1_nor, allFeatures, feature_self_mask, feature_cosp_mask


