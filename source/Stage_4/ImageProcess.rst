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

实时的语义分割网络shuffleSeg
=============================

主要利用分组卷积和通道混洗。
https://mp.weixin.qq.com/s/W2reKR5prcf3_DMp53-2yw
