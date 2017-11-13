******
Paddle
******

http://paddlepaddle.org/index.cn.html

与caffe2非常像，基本上借鉴了caffe,然后解决其不足。
一个很大的不同，采用event-handler 的方式，来进行训练过程中log的输出。

http://book.paddlepaddle.org/index.cn.html,最大的特点是对于 Kubernetes的支持。
对于已经有网络结构做了很好的封装，可以只用添加数据，然后改参数来执行就可以了。

但是扩展起来，就不是那么容易。
http://doc.paddlepaddle.org/develop/doc/howto/dev/new_layer_en.html

可视化与保存
============

serialize/deserialize,to_tar 

