*****
Keras
*****

详细的中文文档。 https://keras-cn.readthedocs.io/en/latest/

#. 建立模型
#. 训练优化
#. 采用己有模型，进行组装适配
#. 进行性能优化分析。 
   
   - 减少少网络尺寸
   - 推理优化，例如编译优化等等。

详细的步骤可以参考 `udemy deeplearning A-Z <https://nvidia.udemy.com/deeplearning/learn/v4/t/lecture/6743752?start=0>`_ 课程目录就是一个很好的template.

tensor的基本操作
================

RepeatVector 类似于 np.stack/concat的功能。
TimeDistributed,主要用一个接口形式的适配，可以用reshape功能来实现
skilearn
========

对于清洗,例如模数转换，例如把各种分类数值化，以及onehotvector等，以及正则化等等都有现成的包。
并且
i

quick start
===========

.. code-block::python

   from keras.models import Sequnetial 
   
   model = Sequnetial()
   model.add(Dense(unit=64,input_dim=100))
   model.add(Activation("relu"))
   model.add(Dense(units=10))
   model.add(Activation("softmax"))
   model.compile(loss="catagorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
   model.fit(x_train,y_train,epochs=5,batch_size=32)
   loss_and_metrics =  model.evulate(x_test,y_test,batch_size=128)
   classess = model.predict(x_test,batch_size=128) 
   

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
   module.summary()



sample 的分析
=============

Addition_RNN
------------

采用了类似于训练诗集一样的方法，对有效的字符 "1234567890+ " 进行编码。 
然后分成两段采用了Seq2Seq的方式来进行。 
https://www.jianshu.com/p/16bbcad00956. 

用到字符进行onehot编码，然后按照XYZ+ABC,7位的长度来进行编码。

基本做到128 HN,能达到99%的正确率，但是实现那个100的突破呢。
#. 
