
#  Instance Segmentation using Mask-RCNN 


## Dataset Used : 
****Ebay's Modanet** Dataset** : Street fashion images dataset consisting of annotations related to RGB images. ModaNet provides multiple polygon annotations for each image. This dataset is described in a technical paper with the title [`ModaNet: A Large-Scale Street Fashion Dataset with Polygon Annotations`](https://arxiv.org/pdf/1807.01394.pdf).

[Dataset Link ](https://github.com/eBay/modanet) 

#### Sample Image :
![enter image description here](https://github.com/shreyjasuja/mask-rcnn-modanet/blob/master/test.png?raw=true)

## Classes : 
Each polygon is associated with a label from 13 meta fashion categories. The annotations are based on images in the [PaperDoll image set](https://github.com/kyamagu/paperdoll/tree/master/data/chictopia), which has only a few hundred images annotated by the superpixel-based tool. 

    1. bag
    2. belt 
    3. boots
    4.footwear
    5. outer
    6. dress 
    7. sunglasses 
    8. pants 
    9. top
    10. shorts
    11. skirt
    12. headwear
    13. scarf/tie

 
## Requirements
Keras 2.1.5, Tensorflow-gpu 1.15, Install Mask_RCNN ([Matterport](https://github.com/matterport/Mask_RCNN))

## Preprocessing 

1. Data augmentations can help in increase the training examples and help to increase the performance of architechture. I have used only `Fliplr(0.5)` (Left and Right flip) using imgaug


2. Since images for validation set were not available, as given in modanet dataset. I took the train dataset json consisting of 52,377 images and did 85 : 15 split it into train json and validation json.
`python split.py -s 0.85 modanet2018_instances_train.json data/train/instances_train.json data/val/instances_val.json`

## Model Used 

   I used [Matterport](https://github.com/matterport/Mask_RCNN)'s implementation of Mask- RCNN
        
Mask R-CNN (regional convolutional neural network) is a two stage framework: the first stage scans the image and generates  proposals(areas likely to contain an object). And the second stage classifies the proposals and generates bounding boxes and masks.

It was introduced last year via the  [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)  to extend its predecessor,  [Faster R-CNN](https://arxiv.org/abs/1506.01497), by the same authors. Faster R-CNN is a popular framework for object detection, and Mask R-CNN extends it with instance segmentation, among other things.

Matterport's Mask RCNN uses a ResNet101 + FPN backbone

I trained only head layers of architecture upto 40 epochs using pre-trained weights on COCO dataset.
  
 #### Modes to Run:
 1. Training  --- `python fashion.py train --dataset data --weights coco`

 2. Inference --- `python fashion.py inference --weights=/path/to/trained/model/weights --image=<file name or URL>`
## Results

    Epoch 40/40 
    100/100 [==============================] - 128s 1s/step -
    loss: 1.2680 - rpn_class_loss: 0.0223 - rpn_bbox_loss: 0.3199 - mrcnn_class_loss: 0.3883 - mrcnn_bbox_loss: 0.2502 - mrcnn_mask_loss: 0.2873 -
    val_loss: 1.2655 - val_rpn_class_loss: 0.0294 - val_rpn_bbox_loss: 0.3628 - val_mrcnn_class_loss: 0.3465 - val_mrcnn_bbox_loss: 0.2408 - val_mrcnn_mask_loss: 0.2860

### Test Images:

Test 1 :Detected -

 - Headwear 
 -  Sunglasess
 -  pants
 -  footwear

![enter image description here](https://github.com/shreyjasuja/mask-rcnn-modanet/blob/master/test1.png?raw=true)

Test 2: Detected -

 - Outerwear
 - pants
 - footwear

![enter image description here](https://github.com/shreyjasuja/mask-rcnn-modanet/blob/master/test2.png?raw=true)


Test 3 : Detected -

 - Outerwear
 - bag
 - belt
 - pants
 - footwear

![enter image description here](https://github.com/shreyjasuja/mask-rcnn-modanet/blob/master/test3.png?raw=true)
