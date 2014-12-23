Linux环境的准备
===============

主要是根据 `caffe install manual <http://caffe.berkeleyvision.org/installation.html>`_ 来进行的。主要库的依赖可以用这个 https://github.com/gwli/StudyNote/blob/master/caffe/install/prepare.py 来完成安装。

nvidia dridver 的安装 
---------------------

#. 从 `官网<http://www.nvidia.com/Download/index.aspx?lang=en-us>`_ 下载最新driver, 

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

   .. code-block:: sh

      $ dpkg -i cuda-repo-ubuntu1404_6.5_amd64.deb

#. 安装cuda toolkit

   .. code-block:: sh

      $ sudo apt-get install -y cuda-toolkit-6-5

#. 配制环境变量
   
   .. code-block:: sh

      $ export PATH+=/usr/local/cuda-6.5/bin/:
      $ LD_LABRARY_PATH+= /usr/local/cuda-6.5/bin/lib:


