******
AI Art
******


#. 用机器模模仿梵高 http://phunter.farbox.com/post/mxnet-tutorial2

如何拼接多张图片
================


.. code-block:: bash
   
   convert *.png -append BigBoy_v.jpg
   convert *.png +append BigBoy_h.jpg
   montage -tile 4x -geometry +2+2 *.jpg montange.jpg
   montage -label '%f' -tile 4x -geometry +2+2 *.jpg montange.jpg
   
label 的format 字符串 see [imagemagick]_

References
==========

.. [imagemagick] http://www.imagemagick.org/script/escape.php

