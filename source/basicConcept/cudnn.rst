cudnn
-----

cuda的deeplearning 库


流程本身与CUDA是一样的,数据格式可以改变access pattern可以尽可能利用cache来提高效率。
kernel本身计算模式的改变可以尽可能利用硬件资源。一个算法本身计算量可以大大的减少。
通过thread，L1,L2,register等等,以及 stream, cdn,cnp资源的分配来提高硬件的occupancy.

.. graphviz::
   
   digraph flow {
       H2D -> "kernel exuection" -> D2H;
   }

目前为止，这是CUDA执行模式。各种库无非就在些基础上来进行进一步限定。

或者添加些help函数来实现实现自动的thread分配。并且把一些算法的kernel来实现一下，
并且更加上一些元编程把kernel的实现做一些自动的适配与优化。thrust都是这样干的。

.. code-block:: c

   checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       srcTensorDesc,
                                       srcData,
                                       filterDesc,
                                       conv.data_d,
                                       convDesc,
                                       algo,
                                       workSpace,
                                       sizeInBytes,
                                       &beta,
                                       dstTensorDesc,
                                       *dstData) );
   

#. Convolution 的kerneldnn 自己实现了。
#. srcTensorDesc,dstTensorDesc, 就是为描述资源的格式与大小方便后存储与对memory
   以及kernel的thread的分配。
#. 主要采用是四维的张量计算
#. 所谓network其实就是一个cudnnCreate 一个cudaContext而己。用完之后是需要cudaDestroy的。
   这样可以把数据保存显存里，然后方便多个kernel之间的计算减少D<=>H之间的传输而己。
   可以使用 CUDA annalysis  对其进行分析。
#. kernel中算法本身需要的参数 


其余那就类似于 CUDA runtimeapi 一样，各种各样的get/set以及createhandle之类的东东了。


