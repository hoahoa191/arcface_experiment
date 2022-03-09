# arcface experiment
pytorch & tensorflow implement of arcface 

This is a tensorflow implementation of paper "[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)". This implementation aims at making both usage of pretrained model and training of your own model easier. 

# TODO List

1. *Train with softmax [done!]*
2. *Model evaluation [done!]*
4. *Get embedding with pretrained model [done!]*
5. *Train with Additive Margin loss [done!]*
6. Backbones    
   7.1 *ResNet [done!]*    
   7.2 **mobleFacenet [todo]**    
7. Losses    
   7.1 *Arcface loss [done!]*    
   7.2 *Cosface loss [done!]*
   
   7.3 *Li Arcface loss [done!]*   
   7.4 **Sphereface loss [todo]**
   7.5 **Magface loss [todo]**    


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
##Training Tips(Continual updates)

During the competition of LFR, we found some useful training tricks for face recognition.

* We tried training model from scratch with ArcFace, but diverged after some epoch. Since the embedding size is smaller than 512. If u want try with smaller embeds(128, 64,...), u should use Pre trained model then finetune before train it.
* In 512-dimensional embedding feature space, it is difficult for the lightweight model to learn the distribution of the features.
* If u can't use large batch size(>128), you should use small learning rate
* If ur system not strong, rescale your dataset (maybe 100id, 100imgs/id), small batch size, then adjust hyper-parameter (s ~ 5-10)
* the optimal setting for m of ArcFace is between 0.45 and 0.5.

##References
1. [InsightFace mxnet](https://github.com/deepinsight/insightface)
2. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
3. [AirFace](https://arxiv.org/pdf/1907.12256.pdf)
4. [Insightface_TF](https://github.com/auroua/InsightFace_TF)

