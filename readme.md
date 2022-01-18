# arcface experiment
pytorch & tensorflow implement of arcface 

# References
https://github.com/deepinsight/insightface

https://github.com/ronghuaiyang/arcface-pytorch/blob/master/README.md

https://github.com/auroua/InsightFace_TF

# InsightFace-tensorflow

This is a tensorflow implementation of paper "[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)". This implementation aims at making both usage of pretrained model and training of your own model easier. Whether you just want to use pretrained model to do face recognition/verification or you want train/finetune your own model, this project can give you a favor. An introduction on face recognition losses can be found [here](https://luckycallor.xyz/20190123/FaceLosses.html)(in Chinese).

The implementation referred to [the official implementation in mxnet](https://github.com/deepinsight/insightface) and [the previous third-party implementation in tensorflow](https://github.com/auroua/InsightFace_TF).

- [InsightFace-tensorflow](#insightface-tensorflow)
  - [TODO List](#todo-list)
  - [Running Environment](#running-environment)
  - [Usage of Pretrained Model](#usage-of-pretrained-model)
    - [Pretrained Model](#pretrained-model)
    - [Model Evaluation](#model-evaluation)
    - [Extract Embedding with Pretrained Model](#extract-embedding-with-pretrained-model)
  - [Train Your Own Model](#train-your-own-model)
    - [Data Prepare](#data-prepare)
    - [Train with Softmax](#train-with-softmax)
    - [Finetune with Softmax](#finetune-with-softmax)

# TODO List

1. *Train with softmax [done!]*
2. *Model evaluation [done!]*
3. *Finetune with softmax [done!]*
4. *Get embedding with pretrained model [done!]*
5. **Train with triplet loss [todo]**
6. **Finetune with triplet loss [todo]**
7. Backbones    
   7.1 *ResNet [done!]*    
   7.2 **ResNeXt [todo]**    
   7.3 **DenseNet [todo]**    
8. Losses    
   8.1 *Arcface loss [done!]*    
   8.2 **Cosface loss [todo]**    
   8.3 **Sphereface loss [todo]**    
   8.4 **Triplet loss [todo]**
9.  **Face detection and alignment [todo]**


## Data Prepare

The official InsightFace project open their training data in the [DataZoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). This data is in mxrec format, you can transform it to tfrecord format with [./data/generateTFRecord.py](https://github.com/luckycallor/InsightFace-tensorflow/blob/master/data/generateTFRecord.py) by the following script:


The directory should have a structure like this:

```
read_dir/
  - id1/
    -- id1_1.jpg
    ...
  - id2/
    -- id2_1.jpg
    ...
  - id3/
    -- id3_1.jpg
    -- id3_2.jpg
    ...
  ...
```
