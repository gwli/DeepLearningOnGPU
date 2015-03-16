ImageNet 当前现状和评价标准
***************************

ImageNet相关数据分析，
评价标准：

在分类过程中，只要一类对应label即可，并且最大判决的概率是正确的label时为正确。

error即是错误在总体中的计数：

.. math::

   error \frac{1}{N}\sum \min d{ij}

在单目标定位时，只有完全正确的标注好目标，并且距离不能超过多少才算成功。

也就是使用算法：

.. math::

   d{ij} \max(d(c{ij},Ci),\min d(b{ij},B{ik}))

.. math::

   Recall(t) \frac{\sum{ij}1[s{ij}\geq t]z{ij}}{N}

.. math::

   Precision(t)\frac{\sum{ij}1[s{ij}\geq t]z{ij}}{\sum{ij}1[s{ij}\geq t]}


另外有算法对边界进行修改：

.. math::

   thr(B)\min(0.5,\frac{wh}{(w10)(h10)})

目标探测

标准：用来惩罚丢失目标的算法，和错误探测的目标。（b{ij},s{ij}）用来表示目标的位置b和目标判别。

历年最成功的算法

ILSVRC2010.

关注的算法：
Fisher vector based methods 
线性SVM。

2013 年：卷积神经网络
dropout technique

2014 年： google
GoogLeNet

使用图像分类作为多目标识别的训练数据。

错误训练原因：

The most common error that an untrained annotator is susceptible to is a failure to consider a relevant class as a possible label because they are unaware of its existence.  Therefore, in evaluating the human accuracy we relied primarily on expert annotators who learned to recognize a large portion of the 1000 ILSVRC classes. During training, the annotators labeled a few hundred validation images for practice and later switched to the test set images.

参考：
ImageNet Large Scale Visual Recognition Challenge

扩展阅读：

PASCAL Visual Object Classes Challenge (VOC) http://www.pascalnetwork.  org/challenges/VOC/voc2012/workshop/index.html. 实现算法评定。

新方向：

Segmentation Propagation in ImageNet  新图像定位方法：使用弱监督定位方法。

Devise: A deep visual-semantic embedding model. large-scale zero-shot learning



