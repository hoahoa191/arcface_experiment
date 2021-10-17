import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot_his(x, y, title, xlabel, ylabel, savepath):
  plt.figure()
  plt.plot(x, y)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(savepath)

def plot_roc(fpr, tpr, path,  label, linestyle='-', c='red', max_acc=None, dtype=None):
  '''
    fpr : array false positive rate
    tpr : array true positive rate
    path: path to save figure
    label: name model
    max_acc: max accuracy 
    dtype : name of test dataset
  '''
  plt.style.use('seaborn')

  plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01) , linestyle='--', color='blue', label='base')
  plt.plot(fpr, tpr, linestyle=linestyle, color=c, label=label)

  plt.title('ROC curve on {}, max accuracy : {}'.format(max_acc, dtype))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive rate')

  plt.legend(loc='best')
  plt.savefig(path,dpi=300)