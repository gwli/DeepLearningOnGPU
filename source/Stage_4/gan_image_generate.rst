****************
各种图像生成算法
****************


为什么GAN会有这么强有力，主要是因为他用一个网络来代替了原来的简单LOSS函数来进么反馈。 一个确定的loss的是很难直观的设计，所以利用网络来代替一个具体的loss函数。一个更高的目标，那就是生成一个稳定可靠的生成式来代替传统的固定的简单生成式。


GAN中主要是利用G来生成实现各种生成。
icGAN 给它不同属性得到不同的表达能力。
GeneGAN 把图像编码成内容与属性，通过交换两张图的属性，实现属生的互转。
Face Age cGAN  对于年龄的翻译。
图像翻译方法的完备性:  content consistency, Domain consistency.
pix2pix_ 实现了从一个域到另一域的任何转换， pix2pixHD_ 实现超分辨，
而 startGAN_ 实现 N->1->N的转换。 用一个模型学习多个领域，并且实现任意两个之间的转换。


有了DL的这种模糊能力，我们就可以解决 vorbose的原理与实际动手之间的差距。

.. image:: /Stage_4/gan_image_generate/gan_compare.png

例如一个基本模型+属性，例如喜怒哀乐，就能生成对应的表情。

图片标题的生成
==============

你可以用 www.captionbot.ai 来自动识别。

ATTNGAN
-------

https://github.com/gwli/AttnGAN

Attenion 输入，一个word features另一个每层local Image features.
如何处理文字与图像feature之间的对应的关系，图像的feature 用CNN 中间层来表示。
而texture,用word2Vector来表示。
一种方式是双方都扩展固定大小相融的尺寸来进行点乘，相当求两个项量的夹角来实现。
另一种方式是直接拼接在一起由网络来学习。
还有一种方法，那生成关联矩阵，然后求网络来学习其参数. 所谓的attention最常用做法那就是softmax+6signama. 相关性常见的做 

.. math:: 
   
   R(C_i,e_i) = (c_i^Te_i)/(||c_i||||e_i||)


Attension Generate network, and DAMSM 用计算fine-grained-text-image matching loss. 


Deep Attention GAN
==================


所谓的attention,就是重点，例如添加一个mask, 就简单的理解那就是kernel的滤波器的就这样的结构，如何把原来滤波器的知识也都用上。

.. image:: /Stage_4/gan_image_generate/da_gan.png

不光要实现set-set的转换，还要实现instance <-> instance的转换，到后面是不是要实现各个层次，每一个层次内部的上下转换。
这与自己之前的思路是一致，即要解决组内距离，又要解决组间距离，如何解决这个呢，当然是通过多层次的discriminator来实现。
discriminator网络比cost函数强的原因，就是在一个原来cost函数之前套上一个网络层，来解决适配问题。

从这里学到，一个对于确定的知识就要用确定的，不确定的就网络部分来模拟学习就像discrminator一样。 

这里生成网络之前，加了一层DAE,变成DAE+G. 与一张大的G有什么区别。内部更加的有序化。

#. floc 是如何编码组织这些位置信息的。
   #. 拿到一个 Feature Map 的图，然后计算其中间位置，这样。
#. 如何界定组间信息与组内信息的。

主要利用的领域
==============

#. 图片的生成
#. 图片的翻译
     
#. 超分辨的实现
#. 表情的合成
#. 动画的生成
#. 电影的自动合成

GAN 论文的汇总 https://www.jiqizhixin.com/articles/2017-04-21-5

pix2pix
=======

这里所谓的创新那就是生成器 从原来的 noise 到图片，现在加入了自身。但是这个不如starGAN来的痛快，直接把noise给扔了.
所谓的 conditionGAN中condition就是指的X。

另外在G生成网络中，采用U-Net解决原来的downsampleing->bottlenecknet->upsampling,这样由于采样信息的缺失，而直接在每一层就像残差网络这样再加自身，加强自身信息的传播，因为本来输入与输出之间都要位置等信息的共享。

.. math:: 
   
   G: {Z} ->Y
   G: {X,Z} ->Y

一个重要的探索不同cost函数与对应特征之间的关系，例如最简单的那L1到L10. 例如l1在pix2pix上会产生模糊。

另外一个重要的应用，那就是原始的那些轮廓图很有可能是直接从激光摄像头或者热成像仪来的。
并且对内容结构的敏感。


它的这个生成模式，其实就是自己想做的，用DL生成已经规律的东东，看看模拟与真实之间关系，
从而打开这个黑盒子。
GAN 可以用来做 迁移学习 :math:`f(\alpha(x,A))=\beta(x,B)`


.. figure:: /Stage_4/gan_image_generate/pix2pix.png

.. figure:: /Stage_4/gan_image_generate/CycleGAN.png

.. figure:: /Stage_4/gan_image_generate/FaderNets.png

见 https://www.jiqizhixin.com/articles/2017-11-03-12

#. G是如何定义
   都采用了convolution-batchNorm-ReLu 的结构.采用了 U-net结构。
   卷积-降采样->bottleneck->upsampling 
   
   .. code-block:: python

      X ------------------------- identity ------------------------------ X
         |-- downsampling ---| submodule | -- upsampling -- | 
      
      class  UnetSkipConnectionBlock:
          def forward(self,x):
              if self.outermost:
                 return self.model(x)
              else :
                 return torch.cat([x,self.model(x)],1)

#. D 是如何定义的
   两种:PatchGAN discriminator/PixelDiscriminator,而这个的优势就利用了PatchGAN分类器来学习结构化的信息。
   区别在不是要用每个像数点来进行最后加权计算，而是每一块的信息来进行加权信息，最后一层的全连接层，前面还是两层结构，最底层不是像数，而是块相当于综合考虑了 texture/style的信息。
   
#. 基本输入信息流是什么
   
   - 原来的图片是合在一起的，直接用Image.open->Transforms.ToTensor 读到Tensor中，然后再把其分开。
     然后根据训练方向来选择，

     .. code-block:: python
        
        def forwoard(self):
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)


并且验证了这些功能

#. Semnatic Labels <-> photo
#. Architectural labels ->photo
#. Map <-> aerial photo
#. BW ->color photo
#. Edges -> Photo
#. Sketch -> photo
#. Day ->night 

这些训练基本就在1/2 hours1 Pascal Titan X GPU.

另外在判别真实器性一方面用AMT用真人来测，另一方面用最新识别系统来进行判别。
例如最新的imagenet测试系统能否认出该物体。

并且去除马塞克也是利用pix2pix来实现，而其中的样本也采用open_nsfw模型来进行打码，然后来进行解码。这些在github都能找到。

pix2pixHD
=========

它解决了，pix2pix 训练不稳定，并且不能生成高分辨的图片，并且生成的图片细节与真实性的问题。

解决了从无尺度变化->label. 

#. 如何解决了多尺度的变化，并且实现不尽可能的真实。
#. 如何解决了训练的不稳定性。
#. G是如何定义
   分层 {G1,G2}, 并且G1 为global Generator,G2为local Generator. 然后G1与G2的连接就如下代码，直接用利用向量格式拼接起来。
   在这一点上有点像capsule网络的向量进向量出的模型。然后逐层训练。只要保证每一层的接口是兼容就可以了。
   G1采用了与StarGAN的结构，conv-residualBlock-conTrans 并且保证与原输入同样的的大小的尺寸。 
   
   .. code-block:: python
      
     
              ### final convolution
              if n == n_local_enhancers:                
                  model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
              
              setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
              setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
          
          self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

      def forward(self, input): 
          ### create input pyramid
          input_downsampled = [input]
          for i in range(self.n_local_enhancers):
              input_downsampled.append(self.downsample(input_downsampled[-1]))

          ### output at coarest level
          output_prev = self.model(input_downsampled[-1])        
          ### build up one layer at a time
          for n_local_enhancers in range(1, self.n_local_enhancers+1):
              model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
              model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
              input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
              output_prev = model_upsample(model_downsample(input_i) + output_prev)
          return output_prev

     
   
#. D 是如何定义的
   也是采用多尺度的方法。  
   input: {instance boundary map,sematic label map,and real/synthesized image}
    
   采用了 setattr+ 前缀编号来实现的。 然后在调用forward时，再拼接起来。
   .. code-block:: bash

      if getIntermFeat:
              for n in range(len(sequence)):
                  setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
          else:
              sequence_stream = []
              for n in range(len(sequence)):
                  sequence_stream += sequence[n]
              self.model = nn.Sequential(*sequence_stream)
       #**

#. 基本输入信息流是什么
   
   - 原来的图片是合在一起的，直接用Image.open->Transforms.ToTensor 读到Tensor中，然后再把其分开。
     然后根据训练方向来选择，

     .. code-block:: python
        
        def forwoard(self):
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)

#. LOSS 函数如何定义
#. 超分辨是如何实现的


对于卷积核的大小的，是与你补捉的细节的程度是相关的。越小，细节越多。

几个小的卷积核叠加在一起，相比一个大的卷积核与原图的连通性不变，但是却大大降低了参数的个数以及计算的复杂度。


Cycle-GAN
=========

实现了图片的翻译，例如黑夜与白天，季节变化的转换，https://junyanz.github.io/CycleGAN/

starGAN
========

创新，如何把多个label合并在一起，并且能够设计出合理g_loss,d_loss来自适应那种label的自由扩展.


如何实现downsampling
--------------------

#. 输入  图像[16,3,128,128] + label[16,5,128,128] = G_input[16,8,128,128]
这个是通过步长来实现
.. math::
   
   O=(W-F+2P)/S+1
   (128-7+2*3)/2+1=64

如何实现up-sampling
-------------------

利用转置卷积，convTranspose2D_ 来实现的其计算公式与上面更相板

i(input)=4, k(kernel_size)=3,p(pading)=0,s(stripe)=1,o(ouput)
.. math::
   
   W = (O-1)*S - 2P + F

这个原理可以参考在实际的实现计算卷积的时候，为了充分利用GEMM来进行计算。

.. math::

   [i,i]=>[i*i,1]
   [k,k]=>[o*o,i*i]   
  
   o =k*i = [o*o,i*i] x [i*i,1]=[o*o,1]


要把整个输入拉成一维的， 然后把 kernel扩展，然后直接用GEMM相乘。同样反过来
推理求i,相当于求逆的过程。


.. math::

   i = o*k'=[i*i,o*o] x [o*o,1]


造成歧义让大家理解的计算方式，与实际的用矩阵计算方式是不一样的。

因为kernel的填充是有规则，是可以按照规则计算出来的。

这样只要保证，conv 同样的输入，就能反算出输入。在这个反算的过程是要求 填充的kernel的逆的。
但实现只是保证了形状的一样，直接使用的转置。 只是保证了矩阵形状的一样。只有正交矩阵的情况下
逆=转置

如何实现 recover
----------------

是交叉实验来实现的。

.. code-block:: python 

   fake_x = self.G(real_x,fake_c)
   rec_x = self.G(fake_x,real_c)

   g_loss = g_loss_fake + self.lambda_rec *g_loss_rec + self.lambda_cls +g_loss_cls

同时多label的训练，其本质就是定义多个loss函数，然后他们求和放在一起训练。

loss 是如何定义
---------------

#. g_loss
#. d_loss
   
   d_loss = d_loss_real + d_loss_fake +self.lambda_cls * d_loss_cls
   d_loss_cls, 来计算标签的cross-entropy, 多值的时候用，binary_cross_entroy_with_logits

optimimzer
----------

优化器使用的的Adam

D网的构成
---------

#. 要判断是不是真图，

#. 要判断这个图的类型 

.. code-block:: python

   # out_src[16,2,2] 真假，来源于哪一个图片集, out_cls[16,5]
   out_src,out_cls = self.D(real_x)
  

bottleneck 有什么用
-------------------

是为了减少计算量，减少参数的个数。 同时采用Resnet来保证网络的深度。



网络拓扑
--------

starGANPaper_ 

.. math::
   
   G(x,c) ->y

.. code-block:: bash

   Generator (
     (main): Sequential (
       (0): Conv2d(8, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
       (1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
       (2): ReLU (inplace)
       (3): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
       (4): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
       (5): ReLU (inplace)
       (6): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
       (7): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
       (8): ReLU (inplace)
       (9): ResidualBlock (
         (main): Sequential (
           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
           (2): ReLU (inplace)
           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
         )
       )
       (10): ResidualBlock (
         (main): Sequential (
           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
           (2): ReLU (inplace)
           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
         )
       )
       (11): ResidualBlock (
         (main): Sequential (
           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
           (2): ReLU (inplace)
           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
         )
       )
       (12): ResidualBlock (
         (main): Sequential (
           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
           (2): ReLU (inplace)
           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
         )
       )
       (13): ResidualBlock (
         (main): Sequential (
           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
           (2): ReLU (inplace)
           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
         )
       )
       (14): ResidualBlock (
         (main): Sequential (
           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
           (2): ReLU (inplace)
           (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
           (4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
         )
       )
       (15): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
       (16): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
       (17): ReLU (inplace)
       (18): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
       (19): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
       (20): ReLU (inplace)
       (21): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
       (22): Tanh ()
     )
   )
   The number of parameters: 8430528
   D
   Discriminator (
     (main): Sequential (
       (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
       (1): LeakyReLU (0.01, inplace)
       (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
       (3): LeakyReLU (0.01, inplace)
       (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
       (5): LeakyReLU (0.01, inplace)
       (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
       (7): LeakyReLU (0.01, inplace)
       (8): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
       (9): LeakyReLU (0.01, inplace)
       (10): Conv2d(1024, 2048, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
       (11): LeakyReLU (0.01, inplace)
     )
     (conv1): Conv2d(2048, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
     (conv2): Conv2d(2048, 5, kernel_size=(2, 2), stride=(1, 1), bias=False)
   )
   The number of parameters: 44762048


主要创新实现在
--------------

把原标签也融进来，当做输入，共同训练。相当于例如把标签1 ->128*128. 相当于每一个像素都对这个标签起了作用了，采用了组合映射的策略。

例如我不知道A与B有什么直接关系，但是A与B放在一起当输入然后扔给神经网络来判定。

核心在 G 的Forward函数

.. code-block:: python

   def forward(self,x,c):
       # replicate spatiitally and concatenate domain information
       # x 16*3*128*128 
       # c 16*5
       
       # [16,5]->[16,5,1,1]->[16,5,128,128]
       c = c.unsqueze(2).unsqueeze(3)
       c = c.expand(c.size(0),c.size(1),x.size(2),x.size(3))
       
       # x & c => [16,8,128,128] 
       x = torch.cat([x,c],dim=1)
       return self.main(x)


如何读取数据建模
================

.. code-block:: python

   from torch.utils.data import Dataset
   from torchvision.datasets import ImageFolder
   from PIL import Image

   class CelebDataset(Dataset):
    
       def __getitem__(self,index):
           if self.mode='train':
               image = Image.open(os.path.join(self.image_path,self.train_filenames[index])
               label = self.train_labels[index]
           else self.mode in ['test']:
               image = Image.open(os.path.join(self.image_path,self.train_filenames[index])
               label = self.test_labels[index]
           return self.transform(image),torch.floatTensor(label)        


Deep Dream的生成原理
====================

实际是就是一个黑盒的迭代测试，来了解功能。 对于特定的分类网络，如果我想
知道他有什么功能，那就是不断测试输入与输出来了解。对于一个深度网络来测试。
就是给其一个独立的图片，不断迭代，反向更新调整输入本身，而不是网络。来看看
其最终生成什么，从而了解网络的功能。
http://www.pytorchtutorial.com/deepdream-pytorch/#i

进一步的玩法，那就是用两张图，一张当控制图，都输入网络，得到其特征，
然后它们重新排列，然后做矩阵乘法，最后选择矩阵乘法里面最大的下标，将这些下标对应的原始图片的特征向量提取出作为新的特征向量就可以了。矩阵相乘的时候进行一下转置可以方便的保存想到尺寸。


Neural Style
=============

类似于DeepDream的原理，三个输入，一个content图，一个style图，另外一个随机噪声，输入到一个预训练好的网络，然后定义两个cost函数，
一个内容评价函数，直接用l2范数，而对于风格采用gram矩阵也就是自身的内积乘积。

把约束函数有意义化，原来单一cost函数用组合优化函数来代替进行得到更加精细的控制 。
例如class space,image pixel space,image feature space.
references
==========

.. _pix2pix: https://github.com/gwli/pix2pix
.. _pix2pixHD: https://github.com/gwli/pix2pixHD
.. _starGAN:  https://github.com/gwli/starGAN
.. _starGANPaper: https://arxiv.org/pdf/1711.09020.pdf
.. _convTranspose2D: http://blog.csdn.net/u014722627/article/details/60574260
