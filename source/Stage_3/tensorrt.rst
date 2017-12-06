********
Tensorrt 
********


.. image:: /Stage_3/tensorrt/DL_Deloy_flow.png

主要用来推理，并且 并且能够层级融合优化。
#. 合并 tensor及layers.
#. 参数的调优

.. image:: /Stage_3/tensorrt/tensorrt_optimize.png


.. image:: /Stage_3/tensorrt/tensorrt_proformance.png

现在已经可以做到7ms的时延了。   

这种编程模式已经进化了，加载了网络图的定义，然后再加载参数，然后执行，这不就是新行的加编程模型了。
   

