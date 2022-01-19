import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print("Detect {} Physical GPUs, {} Logical GPUs.".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e.message)
    else:
        print("\nno GPU")


def plot_his(x, y, title, xlabel, ylabel, savepath):
  plt.figure()
  plt.plot(x, y)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(savepath)

def plot_roc(fpr, tpr, path, linestyle='-', c='red', max_acc=None):
  '''
    fpr : array false positive rate
    tpr : array true positive rate
    path: path to save figure
    max_acc: max accuracy 
  '''
  plt.style.use('seaborn')

  plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01) , linestyle='--', color='blue', label='base')
  plt.plot(fpr, tpr, linestyle=linestyle, color=c)

  plt.title('ROC curve max accuracy : {}'.format(max_acc))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive rate')

  plt.legend(loc='best')
  plt.savefig(path,dpi=300)