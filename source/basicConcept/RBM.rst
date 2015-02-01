能量模型
********

EBM通过对变量的每个配置施加一个有范围限制的能量来捕获变量之间的依赖 关系。EBM有两个主要的任务，一个是推断(Inference)，它主要是在给定观察变量的情况，找到使能量值最小的那些隐变量的配置；另一个是学习(Learning)， 它主要是寻找一个恰当的能量函数，使样本中正确的输入输出的能量 比错误的输入输出的能。

这里的能量最小模型，是不是很多事物都要趋近于最小的。
原来的很多东西都是假设首先知道的，然后建立约束函数，在概率不知道的情况下使用能量函数。


能量模型是一种一般性的模型, 在分布未知情况下用能量来表示概率分布：

.. math::

   p(x) = \frac{{{e^{ - E(x)}}}}{Z}


其中Z为规整因子：

.. math::

   Z=\sum_x e^{-E(x)}

.. math::

   l(\theta,D)=-L(\theta,D)=-\frac{1}{N}\sum_{x^{(i)\in D}\log p(x^{(i)})}

对 :math:`\theta`  求偏导，得到：

.. math::

   \Delta = \frac {\partial l(\theta, D)} {\partial \theta}

在有些情况下我们需要引进一些没有观察到的变量，隐含变量。为了表示所有包含v的隐含变量，表示成能量相加形式：

.. math::

   P(x)=\sum_h P(x,h)=\sum_h \frac{e^{-E(x,h)}}{Z}

并进行求导，但是上式中 :math:`Z=\sum_x e^{-E(x)}` ,对x进行求导非常困难，因此通过采样方法获得近似值。

因此采用玻尔兹曼机实现，但是BM方法计算量非常大，后来使用RBM模型,在层内无联系。
\begin{figure}
  \centering
  \includegraphics[width=8cm]{wangluotuopu}\\
  \caption{玻尔兹曼机和限制玻尔兹曼机}
\end{figure}

限制玻尔兹曼机
===============

限制玻尔兹曼机是一种特殊形式的马尔科夫场，可视层和隐含层不断进行交替迭代，趋于稳定。它输入和输出都是0,1 ，为了方便计算，假设可视层和隐藏层都是满足相互独立，有：

.. math::

   \begin{array}
   p(h|v)= \prod_i p(h_i|v)\\
   p(v|h)= \prod_j p(v_j|h)\\
   \end{array}

在v and h没有概率的情况下，使用能量函数来表示概率变量的概率密度：其中变量v and h的能量函数可以表示为：

.. math::

   E(\bf{v},\bf{h})=-\sum_{i\in pixels}b_iv_i-\sum_{j \in features}b_j h_j -\sum_{i,j}v_i h_j w_{ij} 

RBM中使用两层，首先估计隐藏层的概率，然后用随机概率判断隐藏层发生的概率。 这个有吗？

对应的自由能量函数（就是前面的能量模型（对所有每个分量的能量相加、取log））：

.. math::

   F(v) =b\prime v\sum i-\log\sum hi e^{hi(ci Wiv)}

对上式中的h和v进行求导，得到：

隐藏层计算 decoding：

.. math::

   h_j=\sigma(b_j+\sum_i v_i w_{ij})

可视层计算 encoding：

.. math::

   v_i=\sigma(b_i+\sum_j h_j w_{ij})

用能量函数对 :math:`w_{ij}`  求导，得到：

.. math::

   \Delta w_{ij}=\varepsilon (<v_i h_j>_{data}-<v_i h_j>_{recon})

其中data是原始输入数据，recon是重构的数据，同样的方法更新 :math:`b_i` and  :math:`b_j` 。


Gibbs 采样
==========

Gibbs采样是基于马尔科夫链蒙特卡罗（MCMC）的采样方法， 就是当我们无法得到关于X的联合概率密度P(x)，可以利用条件马尔科夫条件概率密度来逼近。随着采样次数n的增加，随机变量 :math:`[x_1(n),x_2(n),...x_K(n)]` 的概率分布以n的几何级数的速度收敛于X的联合概率密度P(X).

因此在执行时，用一个训练样本初始化可见层的状态 :math:`v_0` ，交替进行如下采样：

.. math::

   h_0\sim P(h|v_0), v_1\sim P(v|h_0),\\
   h_1\sim P(h|v_1), v_2\sim P(v|h_1),\\
   ....,v_{k+1}\sim P(P(v|h_1))


对比散度算法
============

尽管采用GIbbs可以获得联合概率密度，但是对数似然方法，需要较大的采样步数才能保证采样样本服从目标分布，尤其是观测数据的维度比较高时。因此hinton 在2002 提出对比散度方法，只需要k=1步Gibbs 采样便可得到足够的近似。（我想是gibbs采样和梯度方法的结合吧），首先估计出 :math:`P(h_{1j}=1|v_1) and P(v_{2i}=1|h_1)` , 进而产生一个可见层的重构。然后代入能量度函数更新 :math:`w, b`  and  :math:`\alpha` 。

HMC模型
=======

HMC模型采样通过动力学方法来获得Hamiltonian 被定位为自身能量和动力学之和：

.. math::

   H(s,\phi)=E(s)+K(\phi)=E(s)+\frac{1}{2}\sum_i\phi_i^2

其中s位置向量， :math:`\phi` 是速度向量。

HMC通过正则分布采样得到：

.. math::

   p(s,\phi)=\frac{1}{Z}exp(-H(s,\phi))=p(s)p(\phi)

对s and  :math:`\phi`  求导得到：

.. math::

   \frac {ds_i}{dt}=\frac{\partialＨ}{\partial \phi}=\phi_i

.. math::

   \frac {d \phi}{dt}=-\frac{\partial H}{\partial s_i}= -\frac{\partial E}{\partial s_i} 


leap-frog算法首先使用马尔科夫链，蛙跳速度跳过t时刻的位置而得到 :math:`t+0.5\delta t` 的速度值，而位置跳过 :math:`t+0.5\delta t` 的速度值而得到 :math:`t+\delta t` 的位置值。应该是更新速度比较快。详细参考[[http://clzx.cqjtu.edu.cn/Upload/news/20111108215347281.pdf][分子动力学模拟]]


对于有限次采样导致变量有可能是不可逆的。因此通过Metropolis判断accept/reject 概率：

.. math::

   Pacc(\chi,\chi1)=min(1,\frac{exp(-H(s1,\phi1))}{exp(-H(s,\phi))})


参考：
=====

#. http://blog.csdn.net/chlele0105/article/details/17309491}{能量模型(EBM)、限制波尔兹曼机(RBM)}

#. http://blog.csdn.net/mytestmy/article/details/9150213}{深度学习读书笔记之RBM （限制波尔兹曼机}


#. http://blog.sina.com.cn/s/blog_890c6aa301010oks.html}{关于数学,关于Learning的一些问题HMC}

#. http://www.docin.com/p-558753215.html 分子动力学和蒙特卡洛模拟

