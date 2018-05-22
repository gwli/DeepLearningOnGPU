残差网络 
========

http://blog.csdn.net/malefactor/article/details/67637785

为什么有效，Skip Connection, WideNet,FractalNet.

#. Residual不是必须的，只是将ResNet以一种规则的方式展开，侧面证明了REstNet
的Ensemble本质。有没有更好的理论解释

#. Depth 不是最重要的,Block 更重要。如果更好的设计Block. 
   - 如何让网络变的矮胖，而不是瘦长

#. 再把GAN 给加进来，那就更好了。

#. 并且DSSPN提供了一个网络有序化tree的一个很好的例子。

DenseNet
========

把ResNet更向前一步，每一层输入包括前面所层的加权。这样思路自己也想过，就是没有动手验证其效果。
