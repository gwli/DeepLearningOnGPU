************
GAN 对抗网络
************

GAN 最好的应用那就是设计模式中生成模式，可以有各种各样的变种模式。

.. image:: /Stage_2/toplogyStructure/AdversarialNetsFrameworksr.jpg
.. image:: /Stage_2/toplogyStructure/gan.jpg


要同时训练两个网络G,D,并且要网络之间反馈传播。 

先训练D，用真实的数据给D :math:`x ->D ->1` 优化方向概率为1方向。
然后输入由G生成的数据给D :math:`G(z) ->D -> 0` 往零的方向优化。

而在训练G时，如何训练把第二部向1的方向优化得到G的参数，但是G的输入是什么。X再加些噪声，还是只有噪声。或者
还只是其他。G的输入应该y本身。 而最终概率为1/2时。效果最好。也是生成器做到以假乱真了。

所以在train G时  :math:`log(1-D(G(z)))` 最小。 也就是 :math:`d(G(z))` 尽可能为1。  
而train D时，则要求 :math:`D(X)` 尽可能为1。
所以两部为合而为就是如下公式

.. figure:: /Stage_2/toplogyStructure/GAN_Target.png

.. figure:: /Stage_2/toplogyStructure/gan_train_figure.png

.. figure:: /Stage_2/toplogyStructure/gan_train.png
   
   红圈，训练D，D是希望V(G,D) 越大越好，所以是加上梯度(ascending).
   第二步训练G时，V(G,D)越小越好，所以是减去梯度(descending).


这里有一个 toy版本可以用 https://github.com/MashiMaroLjc/learn-GAN/blob/master/code.ipynb


https://github.com/zhangqianhui/AdversarialNetsPapers GAN 资料大全
交叉熵的概念相难理解。

https://github.com/hwalsuklee/tensorflow-generative-model-collections  tensorlfow中的GAN实现。


