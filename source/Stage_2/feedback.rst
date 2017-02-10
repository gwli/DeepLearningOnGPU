Backpropgation
==============

整个就是一个链式求偏导的过程。 :math:`\frac{{\PartialC}{W}}` 另外就是图论中所有路径最短的问题。

#. 从整个学习上说，就是一个偏导函数量。要解决的一个问题包括两方面：第一个是学习速度问题，第二个是防止震荡。目前用的都是基于导数的优化。
#. 受到cost函数影响是约束问题的松和紧。 cost ， activate MSE(最小二乘），线性的max 函数 cross-entropy, sigmoid函数 log-likelihood, softmax函数。
#. 超参数的估计，目前是过拟合产生一个主要原因。
#. 具体采用哪一种组合呢，就看你采用哪一种解析了，如果想用要概率模型就要用softmax组合。

单层神经网络(前向传播)
========================

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

后向传导算法（BP算法）
=====================

每一层采用最小均方误差（LMS）模型，采用梯度下降法得到梯度，进而逐层传播到前向网络中去。

.. math::
 
   \frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}\frac{\partial u}{\partial b}=\delta


因为 :math:`\frac{\partial u}{\partial b}=1`, 所以 :math:`\frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}=\delta`, 得到后向反馈的灵敏度： 

.. math::
 
   \delta^l = (W^{l+1})^T\delta^{l+1}\circ f\prime(u^l)

这个模型在无限次迭代中趋于0，也就是没有价值。


输出层的神经元的灵敏度是不一样的：

.. math::
 
   \delta^L= f\prime(u^L)\circ(y^n-t^n)


神经网络就是利用多层信息进行非线性拟合。

权值更新可以表示为：

.. math::
 
   \frac{\partial E}{\partial W^l}=X^{l-1}(\delta^l)^T

.. math::
 
   \Delta W^l=-\eta\frac{\partial E}{\partial W^l}


就是首先求最后一层的误差，逐步扩展到前一层。

实际中对数据训练都是首先前向传导求出实际输出Op,然后和理想输出做对比。得到对比函数，最后使用后向传导调整权值。

并且这种跨层反馈，并且如何自主联网。

随机梯度下降法
==============

如果一次使用所有数据，那就是batch-gradient-descent. 但是这样对大数据来说，计算就不可形了。
mini-batch的原理，是把矩阵变小，这样不需要一次计算整个输入梯度，只用计算部分。 一次一点的来计算。 大小的选择根据硬件matrix的大小限制来进行选择。
http://neuralnetworksanddeeplearning.com/chap1.html 公式18. 
随机的指的就是那个mini-batch, 正常每次全局的WB来算。 再来计算cost函数，每一次同时算，计算量太大没有办法算，只能每一次算抽取样本来模拟总体cost.
其实就是求平均值的问题，1/3(a+b+c) 与1/2(1/2(a+b)+c) 是不是趋于相当，或者相当于同阶无穷小。
这就是在第5章为什么提到sgd的噪声的原因。

W值的过大，就会出现exploading,过小就会消失。 这个是混沌理论。就是利用混沌理论来设置W值。
难点那就是保证凸就好办，不然很计算全局最优点。

牛顿法比梯度法快的原因http://www.zhihu.com/question/19723347

过拟合与规则化
===============

规则化就是把相当于把先验知识都提前加进去。http://blog.csdn.net/zouxy09/article/details/24971995
http://blog.csdn.net/zouxy09/article/details/24972869
L1就是Lasso,L2就是ridge岭回归。L0产生稀疏，L1是L0最好的近似。


就像背单词一样，训练的迭代次数与就像背单词的记忆是一样的。http://yuedu.163.com/news_reader/#/~/source?id=b7b38304-5450-41eb-8a87-884c98c2336e_1&cid=6281da38266a4cd19fca1c2ae370377e_1
迭代是正道啊。

神经元之间的联系，是通过W值来进行，同时如何反馈关联决定的。

对于学习速度的超参数，可以先大后小，采用可变值，或者加入一个当前的梯度检测，梯度太小时，用大值，梯度小时用大值。 或者加入一个约束项。


过拟合一个原因，参数过多，另一个原因数据不够，就会出方程数少于变量数的多解问题。

特定的细胞只对特定方向的事物感兴趣，一个是利用元胞机来解决专注的问题，然后利用聚类来组网，利用遗传算法来重构网络。

迭代的终止
==========
同时训练的时候，要解决什么时候结束训练的时候，一个简单的就是迭代次数，另外根据Error rate. 达到某个值或者保持某个范围不变之后就停止。过度训练也不好。
http://neuralnetworksanddeeplearning.com/chap3.html



为了加快计算

非导数的学习方法
================

学习的慢，是因为用的导数，如果不用导数呢，不要求连续函数，直接离散呢，就相当于计算机的精度问题，例如参数只能是0，1这样形成了二值网络，就加快了，学习的速度。但是二值网络如何学习呢。
神经网络也是离散的，但是还要保证其光滑的，就是像光栅化的做法一样。
是不是可以用三态门来做，再加一个不定态。

而元胞机以及遗传算法，就不需要导数。还有RNN的LSTM三态门的方法。

另外的初始化的W,B可以根据输入信号本身的特点来取列如公式123.http://neuralnetworksanddeeplearning.com/chap5.html

dropout
=======

相当于人为减少变量的个数，这样可以减少过拟合。 因为计算的问题。 参数要比输入多的多。这样就注定不只有一个解的问题。
