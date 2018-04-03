PSPNet图像的语义分割
====================

.. image:: /Stage_4/ImageProcess/PSPNet.png

是从feature进来之后，同时采用多个不同kernel同时时行采用，然后双线性插值来
保证大小的不变，然后在合并起来，然后再conv生成最后语义图。

其实这个设计最初的原因于caffe的每一层的kernel的大小都必须一致。这样可以简化计算。
因为整个计算都是基于tensor的，所以大小不一样的sensor，拼接合并，就会比较头疼。
语言的分割 MeanIoU/PixelAcc, 44.94/81.69还没有越城人类水平。

Globally And Locally Consistent Image Completion
================================================

.. image:: /Stage_4/ImageProcess/globalAndLocalConsistent.png

http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

这个方法，去除水印，以及补全图片，更加逼真。 它用GAN，但是用两global,local discrimnal. 
来衡量不的标准参数指标，这里提供另一个思路，也就是同时多个目标的优化，并且不需要同时把
两个指标放在同一个cost函数中，可以独立的进行。 

#. Input + Mask -> remove/complete Image. 

当然这个功能用于3D重构，也会不错的效果。
并且这个complete的有效性达到77.0%.的以假乱真的目标。 

实时的形义分割网络shuffleSeg
=============================


IoU 58.3% on CiyteScapes, 15.7 FPS on Jetson-TX2. 
主要利用分组卷积和通道混洗。
https://mp.weixin.qq.com/s/W2reKR5prcf3_DMp53-2yw
LinkNet 语义分割，采用U-Net的思路引入residual Blocks.并且能跑在Jetson-TX1上。
单独使用 group convolution 有损精度，所以channel shuffling来补偿。
其本质就是其尽能独立不相关，包含尽可能多的信息量。

R-CNN,Fast-R-CNN,Mask-R-CNN
===========================

https://www.cnblogs.com/skyfsm/p/6806246.html

.. image:: /Stage_4/ImageProcess/R-CNN.png


R-CNN 原理是把分类与检测结合起来。

#. 如何检测区域,
   直接利用feature层来实现，同时利用loss定义，直接滤波输出boundingbox，以及label的。
#. 如何适应区块大小不一样。大小不一样，最终都缩放到227*227的大小。
#. bounding box是大小如何确定的。
   是通过RPN,从小的合并，然后再NMS得到，每一个记录是x,y,w,h中心位置，以及宽高。 同时需要每某一层需要一个转换那就是到图像原始坐标的转换，这时候就需要训练一个变换来实现这变换。

#. anchor,slideer windows,proposals.
   - ahchor 相当于在不同层之点interest点，不动点，就像SIFT算法的，在不同层同一个点的表示。
   - 在feature 层，输出ROI的大致位置也就bounding box.然后反算回去。

去糊
====

https://mp.weixin.qq.com/s/mtocyqpybSEpwekS20xgmQ
其实是相当于GAN网络，先训练D网，来把先验概率学到，然后再用来去糊。

新型语义分割模型：动态结构化语义传播网络DSSPN
=============================================

https://mp.weixin.qq.com/s/palhFeMnWOZj-T2cqQN7tw

利用动态+树形状态机的模式来实现层次化结构通用分类。 实现跨库的通用训练。采用curriculum learning的划式。 
#. 同时还能把大的网络转化小的网络。例如只顶层的分类的话，就不需要太多的细节。
而子类的只从父的结点输出，进一步分类就行了，不在需要组外的内容，难点是如何实现这种分类。
   - 采用逐级训练的。同时还能用父一层中间状态层。
   - 相当于动态有序的dropout.

YOLO
====

对于 320x320 的图像，YOLOv3 可以达到 22ms 的检测速度，获得 28.2mAP 的性能，与 SSD 的准确率相当但是速度快 3 倍
https://pjreddie.com/darknet/yolo/
