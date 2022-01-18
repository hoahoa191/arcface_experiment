# arcface experiment
pytorch & tensorflow implement of arcface 

This is a tensorflow implementation of paper "[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)". This implementation aims at making both usage of pretrained model and training of your own model easier. 

# References
https://github.com/deepinsight/insightface

https://github.com/ronghuaiyang/arcface-pytorch/blob/master/README.md

https://github.com/auroua/InsightFace_TF


# TODO List

1. *Train with softmax [done!]*
2. *Model evaluation [done!]*
4. *Get embedding with pretrained model [done!]*
5. **Train with Additive Margin loss [todo]**
6. Backbones    
   7.1 *ResNet [done!]*    
   7.2 **ResNeXt [todo]**    
7. Losses    
   7.1 *Arcface loss [done!]*    
   7.2 **Cosface loss [todo]**    
   7.3 **Sphereface loss [todo]**    


## Data Prepare

The official InsightFace project open their training data in the [DataZoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). 

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
