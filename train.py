import os
import tensorflow as tf
import sys
import yaml
import argparse
from tqdm import tqdm

from modules.preprocess import load_tfrecord_dataset
from modules.models import getModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth
from test import validate
from modules.plot import *

######################
#Set GPU
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
set_memory_growth()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    return parser.parse_args()

#####################
@tf.function
def train_step(model, inputs, labels, loss_function, optimizer, **kwargs):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = loss_function(labels, predictions) + tf.reduce_sum(model.losses)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

####################
def main(cofg):
    steps_per_epoch = cofg['sample_num'] // cofg['batch_size']
    save_path = os.path.join(os.getcwd(), "save", cofg['model_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #dataset
    train_dataset = load_tfrecord_dataset(cofg['train_data'], cofg['batch_size'],
                          binary_img=True, shuffle=True, buffer_size=10240)
    #model
    model = getModel(input_shape=(cofg['image_size'], cofg['image_size'], 3),
                     backbone_type=cofg['backbone'],
                     num_classes=cofg['class_num'],
                     head_type=cofg['headtype'],
                     embd_shape=cofg['embd_size'],
                     w_decay=cofg['weight_decay'],
                     training=True, name=cofg['model_name'])
    #restore
    ckpt_path = tf.train.latest_checkpoint(save_path)
    if ckpt_path is not None:
        print("\t[*] load ckpt from {}".format(ckpt_path))
        tf.train.Checkpoint(model).restore(ckpt_path).expect_partial()
        start_step = int(ckpt_path.split('/')[-1].split("-")[-1]) * cofg['step_per_save']
    else:
        print("\t[*] training from scratch.")
        start_step = 1
    #optimizer    
    init_lr = cofg['base_lr'] * (cofg['decay_steps'] ** (start_step // steps_per_epoch ))
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=init_lr, 
                    decay_steps=cofg['decay_steps'], 
                    decay_rate=cofg['decay_rate'])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cofg['momentum'])
    print("\tinitial lr: {}".format(init_lr))

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    compute_loss = SoftmaxLoss()

    #training
    history = {'loss':[], 'acc':[]}
    checkpoint_prefix = os.path.join(save_path, "ckpt")
    train_dataset = iter(train_dataset)
    for step in tqdm(range(start_step, cofg['epoch_num'] * steps_per_epoch + 1)):
        inputs, labels = next(train_dataset)
        loss = train_step(model=model, 
                        inputs=inputs, labels=labels, 
                        loss_function=compute_loss, optimizer=optimizer)
        train_loss.update_state(loss)                          
        if step % cofg['step_per_save'] == 0:
            print("\tsave in step {}".format(step))
            checkpoint.save(file_prefix=checkpoint_prefix)   
        #validation
        if step % steps_per_epoch == 0:
            print("\tvalidate in step: {}".format(step))
            acc = validate(config, dtype='lfw', isplot=False) #need other method
            history['acc'].append(acc)
            history['loss'].append(train_loss.result())
            print("\tEpoch {}===> \tTrain_Loss: {:.5f} Val_acc: {:.5f} \
                ".format(step // steps_per_epoch + 1,  train_loss.result(), acc))
            train_loss.reset_state()
             
    #plot training
    plot_his(range(len(history['loss'])), history['loss'], "Traning Loss",
          xlabel="Epoch", ylabel="Loss",
          savepath=cofg['save_dir'] + "/figures/loss_{}.png".format(cofg['model_name']))
    plot_his(range(len(history['acc'])), history['acc'], "Validation Accuracy",
          xlabel="Epoch", ylabel="Accuracy",
          savepath=cofg['save_dir'] + "/figures/acc_{}.png".format(cofg['model_name']))


if __name__ == "__main__":
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.full_load(file)
    main(config)