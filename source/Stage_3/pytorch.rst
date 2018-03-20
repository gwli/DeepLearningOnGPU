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



工作流程
========

#. 建立网络模型
#. 数据结构建模->tensor
#. 各种tensor之间的适配

   #. torch.squeeze(),torch.unsqueeze()
   #. torch.expand(),torch.cat()

#. 定义优化器。 


tensor基本的数学运算
====================

torch.addcdiv 

.. math::

   out_i = tensor_i + value \times \frac{tensor1_i}{tensor2_i}

torchvision.transforms
======================

常用 image变换，并且chainup.

.. codeblock:: python

   transforms.Compose([ transforms.CenterCrop(10),transforms.ToTensor(),])

quick tutoiral
==============

`PyTorch 中文网 <http://www.pytorchtutorial.com/>`_
.. image:: /Stage_3/pytorch/dynamic_graph.gif


内存管理
========

也就是各种变量的使用，以及作用域。而对于深度网络的框架主要是网络参数的更新，这些参数需要与独立出来，能够加载。

变量
----

基本的数据结构张量(Tensor).

各种运算
========

#. 各种sum的计算，可以看作是group的功能，理解为对某一行或某一例的求和吧。
#. 其他的一些逐点计算，sqrt,还有一些那就是求导计算。
#. 关键是乘法，有多种
   
   - 数乘 (变相的矩阵乘)
   - 内积 也就是标量乘 :math:`a*b = |a| |b| * cos` 就是 :math:`mul-> x_i*y_i`
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

整个网络层的基本结构，参数的存放，然后就是foward与backword的计算。
其他就是一些辅助函数了。就像最基本的类型

.. code-block:: python
   
   class layer :
     def __init__():
         self.W
         self.B
     
     def foward():
         return self.active(self.W*self.X +self.B)    
     def cost():
         error = distance(self.foward(),origina_data_force)

     def backwoard():
         self.W = self.W + xxxxx

这个是网络拓扑的根结构，基本结构也就是dict,并且module是不可以不断嵌入的。

#. addModules 

   code-block:: python
   
   self._modules['module_name'] = module

#. parameters. 这个函数variable的一种封装。因为一个模块的parameter在迭代中才会更新。
当做parameters的变量传给module时，会自动变成其参数的一部分。
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


optim
=====

.. code-block:: python
    
   for input,target in dataset:
        optimizer.zero_grad()
        output=model(input)
        loss = loss_fn(output,target)
        loss.backword()
        optimizer.step()

各种优化算法的原理与区别
------------------------

基本上都是采用的迭代的方法，核心 :math:`\Theta = \Theta - \alpha \cdot \triangledown_\Theta J(\Theta)`

这种方法，容易停在鞍点，

Momentum算法，同时观察历史梯度 :math:`v_{t}`   

.. math::
   
   v_{t} = \gamma \cdot v_{t-1} + \alpha \cdot \triangledown_\Theta J(\Theta)
   \Theta = \Theta -v_{t}

Adagrad
-------

是对learningrate的改变，我们采用频率较低参数采用较大的更新，相反，频率较高的参数采用较小的更新。采用累加之前的所有梯度平方的方法，这个造成训练的中后期，分母上梯度累加变大，就会造成梯度趋近于0，使得训练提前结束。

RMSprop的方法
-------------

采用计算对应的平均值，因此可缓解adagrad算法学习率下降较快的问题。

Adam
----
利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率，使得参数比较平稳。

.. figure:: /Stage_3/pytorch/optims_1.gif
   
   损失平面等高线随时间的变化情况

.. figure:: /Stage_3/pytorch/optims_2.gif

   不同算法在鞍点处的行为比较

http://shuokay.com/2016/06/11/optimization/

有两个超参数beta1,beta2来用计算指数移动均值的时候使用。是不是股票中移动均线的方法来计算。

beta1 : 一阶矩估计的指数衰减率
beta2 ：二阶矩估计的指数衰减率

epsilon: 该参数是非常小的数,主要为防止实现中除零10E-8.

L-BFGS算法
----------

无约束最小化，http://www.hankcs.com/ml/l-bfgs.html，解决了计算海森矩阵的烦恼。但是吃内存，L-BFGS 就是改进内存的使用用的BFGS算法。


how to save the model
=====================

#. torch.save(self.G.state_dict(),"model path")

onnx
=====

.. code-block:: python
   
   from torch.autograd import Variable
   import torch.onnx
   import torchvision
   
   dummy_input = Variable(torch.randn(10,3,224,224)).cuda()
   model = torchvision.models.alexnet(pretrained=True).cuda()
   #。 alexnet.proro caffe 二制model,verbose会打印出可读信息
   torch.onnx.export(model,dummy_input,"alexnet.proto",verbose=True)

   

   #import model in caffe

   model = onnx.load('alexnet.proto')

   #Check that the IR is well formed
   onnx.checker.check_model(model)

   #print a human readdable represnetation of the graph
   onnx.helper.printable_graph(model.graph)


符号推导
========

torch与 numpy不同的地方，很大一部分原因那就是对微分的符号推导的完美实现的。

通过对于一个正常的tensor添加一个wraper,就形成了，就变成了符号，但是又起了Variable的类型，这难免给你一种误解，那就是用蓝色笔来红色的红字。如果把其叫符号，不容易误解了。torch并没有实现符号推导的大部分功能，只实现了微分推导的功能，并且将其推导到了极致。但是它要求推导只能是标量，一次只能对一个Y,对多个x 求导。

一旦把 tensor-> Variable,就是意味着变成符号。但是其类型还保持原来的样子，其实就是利用链式求导法，在v.grad里实现了所有函数操作，abs,+/- tan等等操作微分求法。

由于保存了长链，需要占用大量的内存，所以默认情况下，求导只用一次，并且用完就扔了。如果二次利用就是添加retain_variables=True.

https://sherlockliao.github.io/2017/07/10/backward/, 并且backword之后，可以得到每一个因变量在此处的导数。
而且对于为什么backword可以传参数，就是为解决对tensor对tensor求导的问题。

相当于 先计算l =torch.sum(y*w),然后求l对能够影响到Y的所有变量x的导数。

https://zhuanlan.zhihu.com/p/29923090

.. code-block:: python

   import torch as t
   from torch.autograd import Variable as v

   a = t.ones(2,2)  
   # [1,1]
   # [1,1]
   x = v(a,requried_grad=True)
   # [x00,x01]
   # [x10,x11] 


Step 
====

整个优化迭代就在这个step函数中，

就是实现关键的 :math:`W_n=W_{n-1} + \triangledown\theta`

