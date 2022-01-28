import os
import tensorflow as tf
import sys
import yaml
import argparse
from tqdm import tqdm

from preprocess import load_tfrecord_dataset
from models import getModel
from utils import set_memory_growth
from losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler


######################
#Set GPU
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
set_memory_growth()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument('--m', type=str, default='custome', help='Training mode : Fit or Custome')
    return parser.parse_args()

####################
def main(cofg, mode='custome'):
    steps_per_epoch = cofg['sample_num'] // cofg['batch_size'] + 1
    save_path = os.path.join(os.getcwd(), "save", cofg['model_name']).replace("//", "/")
    log_path = os.path.join(save_path, 'log').replace("//", "/")
    checkpoint_prefix = os.path.join(save_path, "ckpt","w.{epoch:02d}.hdf5").replace("//", "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(log_path)
        os.makedirs(checkpoint_prefix)
    #dataset
    train_dataset = load_tfrecord_dataset(cofg['train_data'], cofg['batch_size'],
                          dtype="train", shuffle=True, buffer_size=10240, transform_img=True)
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
        tf.train.Checkpoint(model).restore(ckpt_path).expect_partial()
        start_epoch = cofg['init_epoch']
    else:
        print("\t[*] training from scratch.")
        start_epoch = 0
    #scale = int(256.0 / cofg['batch_size']) #defaul 512
    #lr_steps = [ scale * s for s in cofg['lr_steps']]
    #lr_values = [v / scale for v in cofg['lr_values']]
    lr_steps = cofg['lr_steps']
    lr_values = cofg['lr_values']
    if cofg['decay']:
        def lr_step_based_decay(epoch):
            lr = cofg['base_lr'] #/ scale
            for i, lr_step in enumerate(lr_steps):
                if epoch >= lr_step // steps_per_epoch:
                    lr = lr_values[i]
            return lr
    else :
        def lr_step_based_decay(epoch):
            return cofg['base_lr']
    optimizer = tf.keras.optimizers.SGD(momentum=cofg['momentum'], clipnorm=1.0)

    #get Loss funtion
    LossFunction=None
    if cofg['loss'].lower() == 'cosloss':
        print("use cosloss")
        LossFunction = CosLoss(num_classes=config['class_num'])
    elif cofg['loss'].lower() == 'arcloss':
        print("use arcloss")
        LossFunction = ArcLoss(num_classes=config['class_num'])
    else:
        LossFunction = SoftmaxLoss()
    if mode != 'fit':
        #graph mode use for debug, trainning maybe faster than fit
        print('custom mode')
        #metric
        metric_loss_train = tf.keras.metrics.Mean(name='train_loss')
        metric_loss_val   = tf.keras.metrics.Mean(name='train_val')
        metric_acc  = tf.keras.metrics.SparseCategoricalAccuracy()
        #checkpoint
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        summary_writer = tf.summary.create_file_writer(log_path)
        train_dataset = iter(train_dataset)
        for step in tqdm(range((start_epoch - 1) * steps_per_epoch, cofg['epoch_num'] * steps_per_epoch)):
            inputs, labels = next(train_dataset)
            loss = train_step(model=model,
                        inputs=inputs, labels=labels,
                        loss_function=LossFunction, optimizer=optimizer)
            metric_loss_train.update_state(loss)

            if step % steps_per_epoch == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'train loss', metric_loss_train.result(), step=step)
                    #tf.summary.scalar(
                     #   'val loss',metric_loss_val.result(), step=step)
                    #tf.summary.scalar(
                     #   'val acc', metric_acc.result(), step=step)
                metric_loss_train.reset_state()
    else:
        print('fit mode')
        callbacks = [ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True, save_freq='epoch', monitor="loss"),
                    TensorBoard(log_dir=log_path),
                    LearningRateScheduler(lr_step_based_decay, verbose=1)]
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=LossFunction, metrics=metrics)

        model.fit(train_dataset, epochs=cofg['epoch_num'],
                initial_epoch=start_epoch,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks)


if __name__ == "__main__":
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config, args.m.lower())