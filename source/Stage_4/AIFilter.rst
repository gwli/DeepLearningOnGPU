AI降噪
======

采用各种合成软件，生成器噪声模型与原声模型，然后合成降噪。
[RTN]_ 用实现DL来实现RTN的编码与解码，同时实现降噪的功能。就是把RTN当做一个autoendcode.

把chanel当做一个通道，误码率就是天然loss函数。根据硬件约束，来用网络生成transimiter and receiver.
然后利用 training来得到设计参数。


.. [RTN] https://github.com/gram-ai/radio-transformer-networks
