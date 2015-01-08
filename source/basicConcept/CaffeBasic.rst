深度学习
========

深度学与前些年神经网络的区别，那就是网络层数要比以前多了。这样的话功能就大大加强了。

网络一组layer来组成，layer 又由一组node来组成，层与层之间联接是由forword与backwrod连接组成。每一层都solver，以及module.

foward 用来把input -> output同时计算出 loss cost,而backword 而是根据loss cost来计算梯度来提取下一步w,b，优画成什么程度为止呢，取决于loss cost. 这个就像牛顿求根法一样，到多少步停止。 并且w,b的初值如何选择。

每一个小的神经网络连接再加上决策树形成大的神经网络，不就实现了脑的功能。

