*******
pytorch
*******

这个是由Facebook 重写了一遍的torch, torch 本身底层用C来实现，通常可以把PyTorch当做numpy的替代。
而torch lib 又分为

.. list-table::

   * - TH = TorcH
   * - THC = TorcH Cuda
   * - THCS = TorcH Cuda Sparse
   * - THCUNN = TorcH CUda Neural Network (see cunn)
   * - THD = TorcH Distributed
   * - THNN = TorcH Neural Network
   * - THS = TorcH Sparse


.. figure:: /Stage_3/pytorch/overall.png

#. tensor的结构与numpy的都是通用互转的。

pytorch 动态指的其使用命令模式，而不是符号式。这样好处在于动态修改网络。这样的好处，就像传统的编码方式，动态性就比较好。并且传统的堆栈式调试都可以派上用场了。

对于 PyTorch 来说，Chainer 是一个先行者。PyTorch 主要受三个框架启发。在 Torch 社区中，Twitter 的某些研究人员构建了名为 Autograd 的辅助软件包，它实际上是基于 Python 社区中的 Autograd。与 Chainer 类似，Autograd 和 Torch Autograd 都采用了一种称为基于磁带（tape-based）的自动分化技术：就是说，你有一个磁带式录音机来记录你所执行的操作，然后它向后重放，来计算你的梯度。这是除了 PyTorch 和 Chainer 之外的任何其他主要框架都不具备的技术。所有其他框架都使用所谓的静态计算图，即用户构建一个图，然后将该图传递给由框架提供的执行引擎，然后框架会提前分析并执行它。

这些是两种不同的技术。基于磁带的差异化使您更容易调试，并为您提供更强大的某些功能（例如，动态神经网络）。基于静态图形的方法使您可以轻松部署到移动设备，更容易部署到异乎寻常的架构，提前执行编译器技术的能力等等。 

并在Facebook内部，PyTorch用于所有研究项目，Caffe2用于所有产品项目。两者模型是可以相互转换的。

.. code-block:: python

   train_set = torch.FloatTensor(training_set)


quick tutoiral
==============

`PyTorch 中文网 <http://www.pytorchtutorial.com/>`_
.. image:: /Stage_3/pytorch/dynamic_graph.gif


framework
=========

#. http://www.pytorchtutorial.com/pytorch-source-code-reading-network/

source code reading
===================

torch.nn.module 的结构就是dict,addmodule,也就是key,value    
