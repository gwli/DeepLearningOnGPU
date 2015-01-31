神经网络基础
************

神经网络起源
============

神经网络起源于对大脑皮层的研究，神纤细胞的轴突末梢（也就是终端）->神经细胞=处理端f(x)。

.. math::

   \begin{figure}[htp!]
     \centering
     \includegraphics[width=6cm]{neutral}\\
       \caption{神经元}
   \end{figure}


大脑特征：
========= 

#. 大脑皮层具有 分布式 并行的特点，
#. 最后的输出具有归纳和推广功能。
#. 神经输出按照激活（fire）和不激活两种选择。

.. math::
   \begin{figure}[htp!]
     \centering
     \includegraphics[width=6cm]{NN.jpg}\\
     \caption{神经网络基本结构}
   \end{figure}

虽然神经网络具有复杂的结构，但是人脑只是简单的模拟，成为人工智能（ANN），利用f(x) 模拟人脑思考中的非线性。

特征表示的粒度：

比如汽车和人脸的基是差别很大的，因此深度学习的根本目标就是找"basis"。

大脑模型：

.. graphviz::

   digraph G {
   rankdir=LR
      
   Memory1->Predict[label="feature1:Color"]
      
   Memory2->Predict [label="feature2:Construct"]
      
   Memory3->Predict [label="feature2:3D information"]
      
   Memory4->Predict [label="feature3:spatial and time seires information"]
      
   Predict->Output
   
   }

*这里的time series 着的是多层之间吗？input？*


单层神经网络{前向传播}

假设C类，N个训练样本的结果。

.. math::
 
  E^N=\frac{1}{2}\sum_{n=1}^{N}\sum_{k=1}^C(t_k^n-y_k^n)^2

这里 :math:`t_k^n` 表示第n个样本对应的标签的第k维。 :math:`y_k^n` 表示第n个样本对应的网络输出的第k 个输出。

对于样本n的误差可以表示为：

.. math::
 
   \begin{array}{l}
        E^n=\frac{1}{2}\sum_{k=1}^C(t_k^n-y_k^n)^2=\frac{1}{2}||\textbf{t}^n-\textbf{y}^n||_2^2\\
        \end{array}

那么l层的误差可以表示为：

.. math::
 
   \begin{array}
    E^n=\frac{1}{2}\sum_{k=1}^C(t_k^n-y_k^n)^2=\frac{1}{2}||\textbf{t}^n-\textbf{y}^n||_2^2\\
   \end{array}


对于传统的神经网络需要计算网络关于每一个权值的偏导数。我们用l表示当前层，那么当前层的输出可以表示为：

.. math::
 
   \begin{array}
   x^l=f(u^l)\\
   s.t.\; u^l =W^lx^{l-1}+b^l
   \end{array}


这里  :math:`x^l` 是下一层的输入，这一层的输出。


输出激活函数 :math:`f(.)` 可以有很多中，一般是sigmoid函数或者双曲线正切函数。意思是把他们进行分类。

.. graphviz:: 

   digraph logistic_regress {
   node [shape = box]
   rankdir=LR;
   {node [shape=circle, style=invis]
   1 2 3 4 5
   }
   { node [shape=point,width=0]
   input
   dummy1
   dummy2
   dummy3
   }
   { rank=same;
   posibity cost
   }
   {1 2 3 4 5}-> input-> function -> posibity -> dummy1 -> prediction -> output [weight=8];
   dummy1->dummy2 [weight=8]
   { rank=same;
   dummy2 -> cost  [splines="ortho"]
   cost -> dummy3 ;
   }
   dummy3-> input [weight=8]
   }




后向传导算法

.. math::
 
   \frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}\frac{\partial u}{\partial b}=\delta


因为 :math:`\frac{\partial u}{\partial b}=1`, 所以 :math:`\frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}=\delta`, 这里bias基的灵敏度是什么意思？？

.. math::
 
   \delta^l = (W^{l+1})^T\delta^{l+1}\circ f\prime(u^l)


这个表示什么意思？这里是基于一个样本？还是多个样本的？ 应该是一个样本的.这个模型在无限次迭代中趋于0，也就是没有价值。


输出层的神经元的灵敏度是不一样的：

.. math::
 
   \delta^L= f\prime(u^L)\circ(y^n-t^n)


神经网络就是利用多层信息进行非线性拟合。

权值更新可以表示为：

.. math::
 
   \frac{\partial E}{\partial W^l}=X^{l-1}(\delta^l)^T

.. math::
 
   \Delta W^l=-\eta\frac{\partial E}{\partial W^l}

反向传导算法：

就是首先求最后一层的误差，逐步扩展到前一层。

实际中对数据训练都是首先前向传导求出实际输出Op,
然和和理想输出做对比。得到对比函数，最后使用后向传导调整权值。

卷积神经网络}

`卷积神经网络: <http://blog.csdn.net/zouxy09/article/details/8775360>`_  

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
   

自动编码}

深度学习读书笔记之 `AE（自动编码） <http://blog.csdn.net/mytestmy/article/details/16918641>`_ 
==============================================================================================================



`深度学习wiki <http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_  

AE对图形不同位置和方向进行边缘检测。另外可用于检测图像隐藏的相关性，和PCA类似。


autoencoders  利用稀疏性来对规则化。


\href{http://deeplearning.net/tutorial/SdA.html#sda}{sda}

只是da的多层堆在一起，每一层算完之后，再整体就像MLP一样计算一遍。autoAE要利用约束防止训练单位阵。

Denoising Autoencoders 原理：

使用code和decode 来求解 :math:`w_{ij}` .

具体如下：

对于输入x建立神经网络：

.. math::
 
   y=s(Wx+b)


其中s是非线性函数：期望得到输出：

.. math::
 
   z=s(W^{T}y+b)


最后使用不同的reconstruction error 作为约束函数：

均方误差（square error ） 和交叉熵

最后使用均方误差作为约束函数：

.. math::
 
   L(x,z)=||x-z||^2


或者使用 `交叉熵(cross-entropy) <http://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5>`_ 作为约束函数：

.. math::
 
   L_H(x,z)=-\sum_{k=1}^d[x_klog{z_k}+(1-x)log(1-z_k)]

square error 只适用于高斯误差，所以cross-entropy 更加鲁棒些。


\Section{Deep Belief Networks深信度网络
}

限制玻尔兹曼机生成获得低层次信息，包含两个层，一个可视层，一个隐藏层，可视层和隐藏层通过吉布斯采样实现，隐藏层的优

L1,L2正则化

我自己的理解就是约束优化函数出现一些没有意义的解。常规的主要L2正则化:

.. math::
 
   J_R(w)=\frac {1}{n}||y-xw||^2+\lambda ||w||^2

但是如果对于高维数据一般存在稀疏性，一般加入L1正则化：

.. math::
 
   J_R(w)=\frac {1}{n}||y-xw||^2+\lambda ||w||^1

2006年tao证明L1正则化等价于0 范数，说明其具有稀疏性。

另外一个是形象性的解释:\href{http://blog.sina.com.cn/s/blog_49b5f5080100af1v.html}{L1 Norm 稀疏性原理}

`KKT 条件 <http://blog.sciencenet.cn/blog-261330-623443.html>`_

.. math::

\min x f(x)

Subject to: g_i(x)\leq 0, h_j(x)=0

这个式子中对于 :math:`g_i(x)` 是一个小于号问题，没有办法写成 求取整体最小值，因此需要转换为对偶问题（在SVM中还会遇到），就是所谓的KKT条件：

1. L(a, b, x)对x求导为零；

2. h(x) =0;

3. a*g(x) = 0;

最后写成:

.. math::
 
   \max_{a,b}L(a,b,x) =L(a,b,x) +a*g(x) +b*h(x)


通过 :math:`max_{a,b}L(a,b,x)` , 只有在 :math:`a*g(x)=0` 的情况下才取最大值。 也就是 :math:`min f(x)`  和 :math:`a*g(x)`  必须是相反的才具有约束意义。*


#. `import gzip 模块 压缩文件 <http://docs.python.org/2/library/gzip.html>`_  
   \href{http://zh.wikipedia.org/wiki/&#37;E6&#37;8B&#37;89&#37;E6&#37;A0&#37;BC&#37;E6&#37;9C&#37;97&#37;E6&#37;97&#37;A5&#37;E4&#37;B9&#37;98&#37;E6&#37;95&#37;B0][拉格朗日乘数}{设置约束函数的时候可以这么干}
#. `LDA-math-MCMC 和 Gibbs Sampling <http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/>`_  
*EM 就是参数估计一种* 把样本值代入直接相乘，把参数当做变量，然后求最大值。前提已经知道了分布。


GIbbs 采样，现在还看不明白

-- Main.GegeZhang - 21 Feb 2014


对比散度（Contrastive Divergence，CD）算法

-- Main.GegeZhang - 21 Feb 2014


判别模型和生成模型，图变换网络(Graph-transformer Networks)，条件随机场，最大化边界马尔科夫网络以及一些流形学习的方法

-- Main.GegeZhang - 21 Feb 2014


自由能量函数

-- Main.GegeZhang - 21 Feb 2014


BM模型结构研究解法

-- Main.GegeZhang - 22 Feb 2014


`LDA-math-MCMC 和 Gibbs Sampling <http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/>`_  gibbs 采样

-- Main.GegeZhang - 22 Feb 2014


对于一些基本的概念是不是应该看？？

-- Main.GegeZhang - 27 Feb 2014


这么多文献时该怎么看？ 只看经典的和新的？

-- Main.GegeZhang - 27 Feb 2014



-- Main.GegeZhang - 15 May 2014


是不是可以借助于tensor 和混合高斯过程来 核函数来求解。


目前问题：


  #. 如何构造每一个感知器，层与层之间如何连接，需要多少层？最简单的方法，每一层之间都是全连接，通过增加层数，来解决所有问题，这样的计算太大。因此如果全联接，要尽可能用剪枝算法，来减少不必要的连接。并且到底需要多少层都是根据实际的情况来的。

      例如数字的分类，最后只有2*2*2种情况，肯定是分不出来的, 所以对于卷积网络，并不是层越多越好。


#. 另外一部分那就是如何反馈，现在看到的都是利用的梯度，建立一个cost函数，然后把所有的参数都放进去，然后求梯度，theano采用链式求导，也就是复合函数求导。只要都是表达式，就可以求导，一次更新所有参数。所以反馈机制，是整体的cost,还是每一层都可以有一个cost,并且反馈采用梯度，还是牛顿法等。

#. 多层之间是可以混合的，例如一层采用卷积，减少到一定程度，然后采用自动编码，最后是隐藏层等。另外神经元之间的横向连接如何建立，也就是层内部关联。


人工智能的未来
===============

大脑是使用记忆来创造的世界，大脑用记忆模型来预测未来，目前的深度学习也体现了这一点。

大脑和计算完全不同，大脑不是靠计算来解决掉问题，而是通过记忆来解决问题。

参考：
=====

\href{http://blog.csdn.net/zouxy09/article/details/9993371}{神经网络基础}
`蜜蜂能够认出你 <http://www.huanqiukexue.com/html/newqqkj/newsm/2014/0409/24296.html>`_  蜜蜂在如此脑容量小的情况下能够认出人脸，有什么启发？

\href{http://freemind.pluskid.org/machine-learning/sparsity-and-some-basics-of-l1-regularization/}{L1,L2 正则化}

\href{http://blog.csdn.net/zouxy09/article/details/8782018}{人工智能的未来}}
