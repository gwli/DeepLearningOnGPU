Keras
*****

详细的中文文档。 https://keras-cn.readthedocs.io/en/latest/


#. 建立模型
#. 训练优化
#. 采用己有模型，进行组装适配
#. 进行性能优化分析。 
   
   - 减少少网络尺寸
   - 推理优化，例如编译优化等等。


基本组成
========

Dense 
   全连接网络 


如何适配
========

本质就是构造矩阵乘来实现。

#. keras.layers.merge.add
#. 残差连接
   
   .. code-block:: python
      
      from keras.layers import Conv2D,Input
      x = Input(shape=(256,256,3))
      y = conv2D(3,(3,3),padding='same')(x)
      z = keras.layers.add([x,y])

快速的可视化
=============

.. code-block:: python

   from keras.utils import plot_model
   plot_model(model,to_file='model.png')
