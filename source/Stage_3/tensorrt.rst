********
Tensorrt 
********

本身又相当于编译器，输入是DL的计算流图加参数，然后要对其进行优化，然后再生成新计算流图，并且也还提供一个相当于ELF engine来能够执行它。

API doc https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/namespacemembers.html

基本概念
========

并且所有layer与Tensor都会有名字，然后通过其名字来查找。

network definition

Layer

Tensor
  tensorrt支持的格式最多8维。最基本的格式N(P_1 P_2 ...) CHW.
  也就是最少的就是NCHW。采用的是按列排序。 最外面是Batch的size.
  最基本的infer.tensor作为输入输出结构。

而看到，GetNetBatchSize,GetNetWidth,就是指这些。

.. image:: /Stage_3/tensorrt/DL_Deloy_flow.png

WorkSpace
   用于指定layer exectuion时，所需要的临时空间大小，并且这个大小layers之间共享的。
   相当于运行之间heap大小。你可以通过 getWorkspaceSize()来得到。

最基本的workflow
================

https://github.com/LitLeo/TensorRT_Tutorial/blob/master/TensorRT_2.1.0_User_Guide.md
使用TensorRT包括两部步骤（1）打开冰箱；（2）把大象装进去：

build阶段，TensorRT进行网络定义、执行优化并生成推理引擎
execution阶段，需要将input和output在GPU上开辟空间并将input传输到GPU上，调用推理接口得到output结果，再将结果拷贝回host端。
build阶段比较耗时间，特别是在嵌入式平台上。所以典型的使用方式就是将build后的引擎序列化（序列化后可写到硬盘里）供以后使用。


#. 读网络定义
#. buildnet
#. optimize
#. provide intput/output. 默认的输入输出已经是host memory了。
#. create engine context
#. do infer
   
   .. code-block:: c
      
      context.execute(batchSize,buffers);
      //or
      context.enqueue(batchSize,buffers,stream,nullptr);

输入输出，都在buffers 里，执行完，直接把buffers中ouputbuffer的Tensor结构copy回去就行了。

.. code-block:: c
   
   //建立一个builder+log 来实现各种framework的转换 
   IBuilder * builder = createInferBuilder(*plogger)
   
   INetworkDefinition  *network = build->createNetwork();
   
主要用来推理，并且 并且能够层级融合优化。
#. 合并 tensor及layers.
#. 参数的调优

.. image:: /Stage_3/tensorrt/tensorrt_optimize.png


.. image:: /Stage_3/tensorrt/tensorrt_proformance.png

现在已经可以做到7ms的时延了。   

这种编程模式已经进化了，加载了网络图的定义，然后再加载参数，然后执行，这不就是新行的加编程模型了。
   

