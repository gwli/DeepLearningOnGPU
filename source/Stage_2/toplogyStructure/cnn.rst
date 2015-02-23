卷积神经网络
============

算法优点：

#. 针对图像中的像素点进行操作，通过卷积和下采样交替进行，在图像分类和识别中有重要应用。

#. 采用感受野和权值共享达到减小隐藏层的目的，同时起到旋转不变的作用。

#. down-sampling 达到减小分辨力的作用，同时也减小运算量。

#. 最后在经过 logistic regression 判断求所有layers的parameters。  %RED% 不难，建立一个cost函数，然后直接梯度计算%ENDCOLOR%


除了卷积网络本身还有什么方法可以来减少的连接数的。

.. math::

   \begin{figure}
     \centering
     \includegraphics[width=4cm]{CNN.jpg}\\
     \caption{卷积神经网络}
   \end{figure}

.. math::
 
   x_j^l = f(\sum_{i\in M_j}x_i^{l-1}*k_{ij}^l+b_j^l)

其中 :math:`M_j` 表示选择的输入maps的集合。（对于图像处理，是获取边缘信息。）

此时的灵敏度可以表示为：

.. math::
 
   \delta_j^l = \beta_j^{l+1}(up(\delta^{l+1})\circ f\prime(u_j^l))

up(.)表示上采样操作。

Sub-sampling Layers 子采样层

.. math::
 
   x_j^l=f(\beta_j^l down (x_j^{l-1})+b_j^l)

其中 :math:`down(.)` 表示下采样函数。

.. graphviz::

    digraph CNN{
   rankdir=LR
   node[shape=box]
   subgraph clusterA {
   
   x_1->y_1 [label="w_11"]
   x_2->y_1  [label="w_21"]
   x_2->y_2  [label="w_22"]
   x_3->y_2  [label="w_32"]
   label="layer1"
   subgraph clusterB {
    y_1
   
   y_2
   label="layer 2 maxpooling"
   }
   }
   y_1->y
   y_2->y
   }
