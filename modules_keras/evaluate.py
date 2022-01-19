import numpy as np
import tensorflow as tf
import os
from modules_keras.preprocess import _transform_images
from modules_keras.utils import plot_roc

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

def evaluate(test_path, pair_file, model, isplot=True, name_plot=""):
    result = {'issame': [], 'prob':[]}
    with open(os.path.join(pair_file)) as src:
        for line in src:
            if line == "" or len(line.split(" ")) < 3 :
                continue

            line = line.split(" ")
            img1 = tf.io.read_file(os.path.join(test_path,line[0]))
            img2 = tf.io.read_file(os.path.join(test_path,line[1]))

            img1 = _transform_images()(tf.image.decode_jpeg(img1, channels=3))
            img2 = _transform_images()(tf.image.decode_jpeg(img2, channels=3))

            batch = tf.Variable([img1, img2])
            embds = model(batch)
            embds = l2_norm(embds)

            diff = np.subtract(embds[0], embds[1])
            dist = np.sum(np.square(diff))

            result['issame'].append(int(line[2])==1)
            result['prob'].append(dist)
    
    thresholds = np.arange(0, 4, 0.01)
    dists = np.array(result['prob'])
    actual_issame = np.array(result['issame']) 
    tpr, fpr, acc, best = calculate_roc(thresholds, dists, actual_issame)
    print("best thres:  {} \t best acc: {}".format(best, np.max(acc)))
    if isplot:
        plot_roc(fpr, tpr, 
        "/home/nhuntn/K64/FaceRecognition/save/figures/{}.png".format(name_plot), max_acc=np.max(acc))
    return np.max(acc)