import numpy as np
import tensorflow as tf
import os
import time
import tqdm

def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

def calculate_accuracy(threshold, dists, actual_issame):
    predict_issame = np.less(dists, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(predict_issame),np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    
    acc = float(tp + tn) / dists.size
    return tpr, fpr, acc


def calculate_roc(thresholds, dists, actual_issame):
    tprs = np.zeros(thresholds.shape)
    fprs = np.zeros(thresholds.shape)
    accuracy = np.zeros(thresholds.shape)

    for i, thres in enumerate(thresholds):
        tprs[i], fprs[i], accuracy[i] = calculate_accuracy(thres, dists, actual_issame)
    
    best_thresholds = thresholds[np.argmax(accuracy)]
    #tpr = np.mean(tprs)
    #fpr = np.mean(fprs)
    return tprs, fprs, accuracy, best_thresholds


def get_featurs(model, dataset, *kwargs):
    result = {'issame': [], 'prob':[]}
    for img1, img2, is_same in tqdm.tqdm(dataset):
        is_same = is_same.numpy() == 1
        
        embds_1 = model(img1)
        embds_2 = model(img2)
        
        embds_1 = l2_norm(embds_1)
        embds_2 = l2_norm(embds_2)

        diff = np.subtract(embds_1, embds_2)
        dist = np.sum(np.square(diff), axis=1)

        result['issame'].extend(list(is_same))
        result['prob'].extend(list(dist))
    
    thresholds = np.arange(0, 4, 0.01)
    dists = np.array(result['prob'])
    actual_issame = np.array(result['issame']) 
    tpr, fpr, acc, best = calculate_roc(thresholds, dists, actual_issame)
    return np.mean(acc)

def evaluate_model(model, dataset):
    s = time.time()
    acc= get_featurs(model, dataset)
    t = time.time() - s
    print('\t--total time is {}'.format(t))
    print('\t--lfw face verification accuracy: ', acc)
    return acc
