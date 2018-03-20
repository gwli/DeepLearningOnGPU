************
Capsules net
************

整个网络实现由Conv实现了建模，而Primate进行最基本的feature抽取变成，而在digits层达到了
整个空间机的构建，而然后又经过三层的全连接，进行重构,来检验digits的效果。并实现n个目标的同时实别。
到digits,是实现单一目标的识别，而到重构层则实现了多个目标的实别。
从0->1-N 的过程。并且GAN的升级版。


机器之心 对于这篇论文 [#R3]_ 进行了解读 [#R1]_ . 其本质核心那就是把原来的二层的网结构。变成三层的网络。

原来网络结构 神经元->层->多层。变成了现在的神经元->Capsules->级->多级。 

这样把原来层与层之间的固定关系连接关系。变成动态的。并且根据一定的协议来连接。

这就给各种不同连接协议了，就像代表了搜索空间找到一条合适的路径。在这些路径还能找到一条更好的路径。

这个相当于神经元的结构更加灵活化，基本上可以看做是向量化面向对象技术，但是这个继承关系是通过类似于
聚类的功能，自动学习出来的。


只需要更少的数据，就能得到更高的识别率，同时能够CNN识别重叠图像的问题。


这个也就进一步融合了传统知识与神经网络。通过Capule 的定义，是不是每一个Capule的都有相对明确的物理意义了，而层与层之间的连接是通过动态学习出来，而不像之前的一切都是无明确意义的参数而己。把Capsule当做一个向量化的对象或者结构体可能更恰当。
并且层层与连接也都是动态学习。出来。

.. image:: /Stage_2/capsulenet/routingAlgorithm.png


[#R2]_ 机器之芯给出一个实现。

.. figure:: /Stage_2/capsulenet/capsule.png
   
   from https://www.jiqizhixin.com/articles/2017-11-05

其实相当于cnn的两层合成了一层。把Pooling层当做一个activate函数了。这样就把原来神经元结构给泛化了。
为什么激活函数要生成一个数值呢，其实就当于高维一点，在低维空间里一个面或者线。用点线面体理论来解决可能会更好。
主要是原来的激活函数只能是一个数值，现在pooling的输出可以是一个向量了. 这个不正是聚类抽象的过程。
实现了低维向高维转化的通路。所以激活函数，本质就是抽象压缩的一个过程。只不过最初的形式太单一了。给不出
很好的物理意义。进一步pooling再到capsule这一步，就体现了抽象压缩的过程。非线性那就是那质变的过程。
现在终于抽象找到了一个很好的数学定义了，抽象的过程，可以用那些非线性过程来定义。而一些统计函数本身就是
很不错的非线性函数模型。

而capsule模型可以替代了残差网络了。

关键是在层与层的连接加入动态协商的过程，正因这个机制加入使利用先验知识能为了可能。
有点类似于AlphaGo的多张网，网与网之间的连接。

如何实现高低维的转换就是在激活函数上实现的。直接把投影变换当做激活函数。这样就可以实现立体实识别。

另换一种那就是上下层之间的连接，我们可以利用相关性。

而每一层capsule层的关系，就像贝叶斯理论来指导。总合其为1。


实现流程
========


#. 大的工作流程

   .. image:: /Stage_2/capsulenet/workflow_1.png
   .. image:: /Stage_2/capsulenet/workflow_2.png



.. code-block:: bash

   root@48d5c7680565:/opt/pytorch/pytorch-capsule# CapsuleNetwork (
     (conv1): CapsuleConvLayer (
       (conv0): Conv2d(1, 256, kernel_size=(9, 9), stride=(1, 1), bias=False)
       (relu): ReLU (inplace)
     )
     (primary): CapsuleLayer (
       (unit_0): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_1): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_2): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_3): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_4): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_5): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_6): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
       (unit_7): ConvUnit (
         (conv0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2), bias=False)
       )
     )
     (digits): CapsuleLayer (
     )
     (reconstruct0): Linear (160 -> 522)
     (reconstruct1): Linear (522 -> 1176)
     (reconstruct2): Linear (1176 -> 784)
     (relu): ReLU (inplace)
     (sigmoid): Sigmoid ()
   )


#. 数据流
   
   - [128,1,28,80]

      .. graphviz:: 
         
         digraph Image {
           graph [layout=dot rankdir=LR,labeljust=1]
           node [shape=record,penwdith=2,fontsize=25];
           subgraph cluster_input {
              label = "batch:1-128";
              fontsize = 40;
              bgcolor="purple:pink";
               subgraph cluster_chanel {
                   label = "channel:1";
                   bgcolor = "blue:cyan";
                   image[label="28*28" fillcolor="red:yellow",style="filled"];
               }
           }

         } 


    -  经过第一层的conv+relu之后，256 kernel, 就形成了。 [128,256,20,20]
       
    -  然后进入primate layer. 然后是这个[128,256,20,20]进入8个并行的，并且每一个unit有32kernel. 然后再把这些kernel squash.
       8个[128,32,6,6], [batch,channel,width,height] -> [batch,unit,channel,width,height] [128,8,32,6,6] 然后再压平变成[batch,unit,features]
    
    -  Squash 就是在这些features 这个来做。    
    -  Digists层，相当于10个onehot vector,每一个向量具有16维，而后面的全连接，则是其参数矩阵。
       来解决一个还是二个的映射组合。
   
这个就像人的认识过程，先做一个预处理，从大量的重复出得到pattern,然后这些pattern最小化。
Primary Caspsules 相当于是经验，一些先验知识与元认知。
    
       
#. squish 的图形
   
   .. image:: /Stage_2/capsulenet/squash.png 
   .. image:: /Stage_2/capsulenet/squash_wolfram.png

#. 而累积的过程，就是靠内积空间来实现的。方向越近，值越大。不相关时是垂直。结果为0。

反馈网络
--------

#. W 的更新

进一步的实验
============

#. 路由信息
   - 改用把softmax 改为六siga原则，取前六，
   - 采用遗传算法，取前几，后面强制零0，每一次的混合都加入随机。
   - 改用EM应该是已经有人在用了。
#. 把压缩变换成PCA呢。
   - 用PCA进行判别。

#. 参数矩阵，不为方阵。是不是就意味着一定有冗余。是不是可以进一步化简得到一个方阵的约束。


通过扭曲空间来执行数据分类，基于向量场的新型神经网络
====================================================

https://mp.weixin.qq.com/s/lwvRz1jtv1aCBwL9W447_A

在这个篇论文中利用场论来构造神经网络，把场函数当做一个激活函数。这样就变成了一个扭曲的场的解释，就找到种现成的数据理论来进行解释了。

其实只要任何一个差分方程，微分方程的迭代式求解都可以变成一个网络结构。W就是参数，激活函数就是基本项。 
cost函数变成对抗网络中discrimiator可以变成 条件约束。例如把KKT等等约束都变成一个cost函数。 

并且深度网络层数就相当于空间的维度，这样的我们可以充分利用高维空间降维，各种理论的可解释性来理解深度网络 。

另外，高维空间里，简单线性算法的潜力，还是非常大的，比如线性的LPP算法，竟然可以获得和非线性manufold learning类似的效果，也是稀奇了。DNN也可以看成是多个LPP搭起来的，加个限幅函数来减少小概率事件影响


Reference
=========

.. [#R1] https://www.jiqizhixin.com/articles/2017-10-28-4
.. [#R2] https://www.jiqizhixin.com/articles/2017-11-05
.. [#R3] https://arxiv.org/pdf/1710.09829.pdf
