************
Capsules net
************

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


Reference
=========

.. [#R1] https://www.jiqizhixin.com/articles/2017-10-28-4
.. [#R2] https://www.jiqizhixin.com/articles/2017-11-05
.. [#R3] https://arxiv.org/pdf/1710.09829.pdf
