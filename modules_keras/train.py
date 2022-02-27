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
def main(cfg, mode='custome'):
    steps_per_epoch = cfg['sample_num'] // cfg['batch_size'] + 1
    save_path = os.path.join(os.getcwd(), "save", cfg['model_name']).replace("//", "/")
    log_path = os.path.join(save_path, 'log').replace("//", "/")
    checkpoint_prefix = os.path.join(save_path, "ckpt").replace("//", "/")
    if not os.path.exists(save_path):
        os.makedirs(log_path)
        os.makedirs(checkpoint_prefix)
    #dataset
    train_dataset = load_tfrecord_dataset(cfg['train_data'], cfg['batch_size'],
                          dtype="train", shuffle=True, buffer_size=10240, transform_img=True)
    try:
        valid_dataset = load_tfrecord_dataset(cfg['valid_data'], cfg['batch_size'],
                          dtype="valid", shuffle=True, buffer_size=10240, transform_img=True)
    except:
        valid_dataset = None
        
    #model
    model = getModel(input_shape=(cfg['image_size'], cfg['image_size'], 3),
                     backbone_type=cfg['backbone'],
                     num_classes=cfg['class_num'],
                     head_type=cfg['headtype'],
                     embd_shape=cfg['embd_size'],
                     w_decay=cfg['weight_decay'],
                     training=True, name=cfg['model_name'])
    #restore
    ckpt_paths = [f.path for f in os.scandir(checkpoint_prefix)]
    if len(ckpt_paths) > 0:
        model.load_weights(ckpt_paths[-1], by_name=True, skip_mismatch=True)
        start_epoch = cfg['init_epoch']
    else:
        print("\t[*] training from scratch.")
        start_epoch = 0
        
    scale = int(512 / cfg['batch_size'])
    lr_steps = [ scale * s for s in cfg['lr_steps']]
    #lr_values = [v / scale for v in cfg['lr_values']]
    lr_values = cfg['lr_values']
    print("lr_steps: ", lr_steps, "\tlr_values: ", lr_values)
    if cfg['decay']:
        def lr_step_based_decay(epoch):
            lr = cfg['base_lr'] #/ scale
            for i, lr_step in enumerate(lr_steps):
                if epoch >= lr_step:
                    lr = lr_values[i]
            return lr
    else :
        def lr_step_based_decay(epoch):
            return cfg['base_lr']
    
    if cfg['optimizer'].lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(momentum=cfg['momentum'], nesterov=True)
    else : optimizer = tf.keras.optimizers.Adam()
    
    #get Loss funtion
    LossFunction=None
    if cfg['loss'].lower() == 'cosloss':
        print("use cosloss")
        LossFunction = CosLoss(num_classes=config['class_num'])
    elif cfg['loss'].lower() == 'arcloss':
        print("use arcloss")
        LossFunction = ArcLoss(num_classes=config['class_num'], margin=cfg["logits_margin"], logits_scale=cfg["logits_scale"])
    else:
        LossFunction = SoftmaxLoss()
        print("use softmax")
        
    if mode != 'fit':
        #graph mode use for debug, trainning maybe faster than fit
        print('custom mode')
        #metric
        metric_loss_train = tf.keras.metrics.Mean(name='train_loss')
        metric_loss_val   = tf.keras.metrics.Mean(name='train_val')

        #checkpoint
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        summary_writer = tf.summary.create_file_writer(log_path)
        train_dataset = iter(train_dataset)
        
        for step in tqdm(range((start_epoch - 1) * steps_per_epoch, cfg['epoch_num'] * steps_per_epoch)):
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
        callbacks = [ModelCheckpoint(filepath=os.path.join(checkpoint_prefix, "w.{epoch:02d}-{loss:.2f}.hdf5"), save_weights_only=True, save_freq=cfg['step_per_save'], monitor="loss"),
                    TensorBoard(log_dir=log_path),
                    LearningRateScheduler(lr_step_based_decay, verbose=1)]
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=LossFunction, metrics=metrics)

        model.fit(train_dataset, #validation_data=valid_dataset, 
                epochs=cfg['epoch_num'],
                initial_epoch=start_epoch,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks)


if __name__ == "__main__":
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config, args.m.lower())