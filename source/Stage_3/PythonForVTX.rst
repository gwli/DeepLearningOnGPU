Pipeline
========


.. graphviz::

   digraph mayavi {
       size = "4,4";
       engine -> scenes -> {"data sources";filters;modules;render;};
   }
   

这个pipeline设计是非常好的，简化了各个部分的开发，基本上与opengl的pipeline类似。data source 来与各种数据格式来打交道，就像图像的各种文件格式一样。但最终形式都是些基本类。

并且在文档方面提供了一个类似于tcl/tk demo 的demo库，在pythonxy/doc/ets 下面以及mayavi里面。那里面的文档还是非常全面的。并且还有一个与expect一样的功能那是自动录制脚本。同时还支持回调事件。可以做交互式的开发。并且还有一个visual的动画库。

.. csv-table:: 

   set_engine , set the mlab engine ,
   figure(),figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(400, 350)),
   gcf() , get current figure ,
   draw() , refreshing function ,
   savefig() ,  ,
   clf() , cleared ,

Source 对于数据类型的划分。一个图是用点还线段，还是grid来表示数据点，这些基本点都是有其坐标的。默认把其下标当作坐标。而其要表示的内容，可以是标量例如只是一个点的大小，颜色。还可以矢量，例如速度，场强等。同时也可混合，点的大小，颜色，速度等等。当然还可以重力等，这个是根据建模对象来，你可以用颜色，大小，以及各种箭头，各种动画来进行标识。再另外的那就是体纹理的绘制。最常用的颜色映射，已经有现成的映射表了。一般需求都可以满足。
#. specify the points of the space
#. Specify the connectivity between the points. 这个区分是显式还是隐式拓扑。
#. The connectivity lets you build "cells" that break the space into pieces.(类似于光栅化之后的事情）
#. specify "attribute" data at the points or cells.

.. csv-table:: 

   array2d_source ,
   builtin_image ,
   builtin_surface ,
   acho_file ,
   grid_source ,
   line_source ,
   point_load ,
   scalar_field , 
   vector_field ,
   probe_data ,


camera 指是观察角度的可以像opengl中那样感变move,roll,view等改变其位置。

.. csv-table:: 

   sync_camera() ,

render display
==============


.. csv-table:: 

   imshow , View a 2D array as an image ,
   surf , ,
   contour_surf , ,
   coutour3d , 相当于体纹理绘制,
   mesh , 这个其实类似于opengl的四边形图元，不过是目标点应该是每一个风格的中心。 ,
   triangular_mesh ,  类似于opengl三角形图元，坐标是顶点, 
   barchar , ,
   quiver3d , ,
   flow , ,
   plot3d ,,

annim 效果可以通过改变数据实现，就像放电影一样，但是这个framerate是如何定义的。并且更新同步的机制是怎样的，是不是也会有像opengl，glflush,glclear,glswap之类的函数来进行同步。

See also
========

#. `mayavi-doesnt-run-from-within-spyder-complains-about-valueerror-api-qstring <http://stackoverflow.com/questions/12442938/mayavi-doesnt-run-from-within-spyder-complains-about-valueerror-api-qstring>`_  change from Qt4 to wx 
   我自己试一下，直接在API选择API#2，并且ignore API change errors. `原因在此 <http://stackoverflow.com/questions/6238193/pyqt-new-api-with-python-2>`_ 
#. `meshgrid <http://baike.baidu.com/view/4430765.htm>`_  为什么要meshgrid的原因
#. `classvtkRectilinearGrid.html <http://www.vtk.org/doc/nightly/html/classvtkRectilinearGrid.html>`_  一个更重要，那就是vtk是由doxygen在线的文档，非常方便
#. `CDASH,ParaView <http://www.vtk.org/pdf/file-formats.pdf>`_  

Thinking
========



*Enthought* 提供了插件机制，这样对于开发一个大的系统是非常有用的，但是在什么时候，需要加入这个功能吗，是需要开始设计功能吗，应该不是，在接口什么都没有确定的情况下，写框架是没有意义的，即是框架都是提供而来的。而不是设计出来的。

-- Main.GangweiLi - 23 Aug 2013


*Traits* 功能就是参数检查与绑定的功能，并且能够可以可视化来进行配置。用起来也非常简单，那就是只需要继承hastraits,并且成员变量当做静态变量来声明，并且python支持多重继承。

-- Main.GangweiLi - 23 Aug 2013


mayavi is a standalone application just like the expect which use tcl as built-in interpreter.

-- Main.GangweiLi - 23 Aug 2013


*python -wthread* 同时进行交互式就需要多线程来支持。单线程是没有办法实现的。要么就得多进程。

-- Main.GangweiLi - 24 Aug 2013


*为什么3D可视化难*
3维需要更多的信息，同时还指定拓扑结构。而这些在二维则是相对的简单很多。

-- Main.GangweiLi - 24 Aug 2013


*vtk/TVTK pipeline*

.. graphviz::

   digraph vtk {
     Source -> Filter -> Mapper -> Actor ->Display;
   }
   


-- Main.GangweiLi - 24 Aug 2013



-- Main.GangweiLi - 19 Aug 2013

