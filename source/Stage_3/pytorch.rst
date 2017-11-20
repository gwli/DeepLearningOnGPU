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


各种运算
========

#. 各种sum的计算，可以看作是group的功能，理解为对某一行或某一例的求和吧。
#. 其他的一些逐点计算，sqrt,还有一些那就是求导计算。
#. 关键是乘法，有多种
   
   - 数乘 (变相的矩阵乘)
   - 内积 也就是标量乘 :math:`a*b = |a| |b| * cos`
   - 外积 也就是点乘。 * 
   - 混合积 
   - 范数

n对于n的数乘就是矩阵乘。

对于矩阵试编程中，随时变量实现。例如 

.. code-block:: python
   
   temp = a*b
   c = temp * 1
   d = temp * 1.5 
   e = temp * 2
   # implment in matrix mul
   matrix_base = torch.stack[[temp] * 3,dim=0]
   matrix_c = torch.sensor([c,d,e])
   matrix = matrix_base *matrix_c

framework
=========

#. http://www.pytorchtutorial.com/pytorch-source-code-reading-network/

source code reading
===================


torch.nn.module 
===============

这个是网络拓扑的根结构，基本结构也就是dict,并且module是不可以不断嵌入的。

#. addModules 

   code-block:: python
   
   self._modules['module_name'] = module

#. parameters.
核心是 __init__ 在这里，生成网络。

#. 然后是其forward函数。需要自己实现。

#. 其核心那就是那个__call__ 的实现。
   
   .. code-block:: python

      def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, Variable):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, Variable)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
      return result

     
