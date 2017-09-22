torch
******

torch 采用 lua 来做其脚本，lua 短小精炼，集C的语法，python的缩进语法，javascript似的函数闭包。


包管理用 *Luarocks* 例如
i
.. code-block:: bash

   luarocks install  XXXX



傻瓜式安装，在运行的时候，可能会遇到GPU driver的匹配问题，重启一下就好了。同时由于
torch 最新支持的cudnn5. 现在机器上的cudnn6. 两个版本改动不大。 直接发把库名与版本号检查改大就行了。

.. code-block::  lua
   :emphasize-lines: 8-10,19-20

   # install/share/lua/5.1/cudnn/ffi.lua
   local CUDNN_PATH = os.getenv('CUDNN_PATH')
   if CUDNN_PATH then
       io.stderr:write('Found Environment variable CUDNN_PATH = ' .. CUDNN_PATH)
       cudnn.C = ffi.load(CUDNN_PATH)
   else
   
       -- local libnames = {'libcudnn.so.5', 'libcudnn.5.dylib', 'cudnn64_5.dll'}
       local libnames = {'libcudnn.so.6', 'libcudnn.6.dylib', 'cudnn64_6.dll'}
       local ok = false
       for i=1,#libnames do
           ok = pcall(function () cudnn.C = ffi.load(libnames[i]) end)
           if ok then break; end
       end

    ------------------ 
    -- check cuDNN version
    cudnn.version = tonumber(cudnn.C.cudnnGetVersion())
    --if cudnn.version < 5005 or cudnn.version >= 6000 then
    if cudnn.version < 5005 or cudnn.version >= 7000 then
       error('These bindings are for CUDNN 5.x (5005 <= cudnn.version > 7000) , '
            .. 'while the loaded CuDNN is version: ' .. cudnn.version
               .. '  \nAre you using an older or newer version of CuDNN?')
    end   

