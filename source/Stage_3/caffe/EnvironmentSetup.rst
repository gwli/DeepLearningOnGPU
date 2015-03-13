Linux环境的准备
===============

主要是根据 `caffe install manual <http://caffe.berkeleyvision.org/installation.html>`_ 来进行的。主要库的依赖可以用这个 https://github.com/gwli/StudyNote/blob/master/caffe/install/prepare.py 来完成安装。

nvidia driver 的安装 
---------------------

#. 从 `官网 <http://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下载最新driver, 

#. 进入字符介面  tty1
   
   :command:`ctl+alt+F1` 

#. 关掉 X windows.

   :command:`sudo service lightdm stop`

#. 安装driver.

   :command:`chmod +x installer.run;./installer.run`

#. 恢复 X windows.
    
   :command:`sudo service lightdm start`

.. note:: 

   正常的情况下installer会自动disable的自带的driver,如果失败的重起一下，重新执行installer就行了。

安装cuda-toolkit
----------------

这里从两种方式来安装，手动安装，或者使用 `jetpack pro <http://developer.download.nvidia.com/embedded/jetson/TK1/docs/2_GetStart/Jeston_TK1_QuickStartGuide.pdf#nameddest=Flashing%20Jetson>`_ 只安装 host端的cuda-toolkit.

#. 添加源

   .. code-block:: bash

      $ dpkg -i cuda-repo-ubuntu1404_6.5_amd64.deb

#. 安装 cuda toolkit

   .. code-block:: bash

      $ sudo apt-get install -y cuda-toolkit-6-5

#. 配制环境变量
   
   .. code-block:: bash

      $ export PATH+=:/usr/local/cuda-6.5/bin/:
      $ LD_LIBRARY_PATH+= /usr/local/cuda-6.5/bin/lib:
      $ export LD_LIBRARY_PATH

    .. note:: 

       也可以直接写.bashrc中就行，重启shell就行了。
   


相关库的安装
------------

直接使用 https://github.com/gwli/StudyNote/blob/master/caffe/install/prepare.py 来完成就行了。

下面几个不常见库的说明

.. csv-table:: libs 
   :header: "Lib names",Content, Remark

   google-glog, google提供的一个C++的logging库, https://code.google.com/p/googl有e-glog/
   protobuf-devel,性能更好的jason用于进程通信,相当于TCP/IP的包解析技术用在了进程通信IPC.
   gflags, 类似于getopt命令行处理接口,http://dreamrunner.org/blog/2014/03/09/gflags-jian-ming-shi-yong/
   google snappy, 一个高速的压缩库，http://www.infoq.com/cn/news/2011/04/Snappy
   google leveldb, LevelDB is a fast key-value storage library written at Google that provides an ordered mapping from string keys to string values.，https://github.com/google/leveldb


安装cudnn 
---------

#. 从 ·https://developer.nvidia.com/cuDNN 下载 linux 官

#. 解压安装

   .. code-block:: bash
      
      $ tar -xzvf cudnn.tgz
      $ mv cudnn /usr/local/cudnn

#. 修改LD_LIBRARY_PATH
      
   .. code-block:: bash

      LD_LIBRARY_PATH+=:/usr/local/cudnn:
      C_INCLUDE_PATH+=:/usr/local/cudnn:
      CPLUS_INCLUDE_PATH+=:/usr/local/cudnn:

      PATH+=:/usr/local/cuda-6.5/bin:
      LD_LIBRARY_PATH+=:/usr/local/cuda-6.5/lib64:
      LD_LIBRARY_PATH+=:/usr/local/cuda/cudnn:
      export LD_LIBRARY_PATH
      C_INCLUDE_PATH+=:/usr/local/cuda/cudnn:
      CPLUS_INCLUDE_PATH+=:/usr/local/cuda/cudnn:

    由于采用 cudnn 的库与caffe兼容性问题，现在 caffe 与V1 是工作的。
    
    一个快速灵活的办法，那就是link.
    
    .. code-block:: bash
       
       ln -s  /usr/local/cudnn-6.5-linux-x64-v2-rc2/ /usr/local/cuda/cudnn
     

build 
-----
caffe 采用了大量的template,所以代码模式基本

.. graphviz::
   
   digraph flow {

       template->code->"PIC so"-> "link to app";
   }



#. build the app
   
   :command:`make all |tee make.log`

#. build the test

   :command:`make test |tee test.log`

#. run the test

   :command:`make runtest |tee runtest.log`


.. note::

   cudnn-6.5-linux-R2-RC1 is compatible with caffe, cudnn-6.5-linux-R1 is good.
   可以在这个https://groups.google.com/forum/#!forum/caffe-users 里找到这个问题

support python
==============

#. 安装依赖

   .. code-block:: bash
      
      cd $caffe_root/python
      for req in $(cat requirements.txt);do pip install $req; done

      

#. installl scipy

 there is miss requirment scipy
 this need fortrain compiler
 you can install it by 
 
   #. install gfortran :command:`apt-get install gfortran-4.8`
   #. make link due can't find gfortran  `ln -s /usr/bin/gfortran-4.8 /usr/bin/gfortan`

   最简单的办法

   :command:`apt-get install python-scipy`

#. 添加caffe to python lib

   .. code-block:: bash
      
      export PYTHONPATH=/$caffe_root/python:$PYTHONPATH

#. 生成文档

   #. install jekyll
      
      :command:`apt-get install jekyll doxygen`
