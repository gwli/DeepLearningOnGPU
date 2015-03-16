.. graphviz::

   digraph{
   
     threads-> warp  label=['32 个独立线程合成最小独立并行单元']
     warps-> block  label=['同一个block中使用shared memory，不同的block中使用global memory']
     blocks-> kernels  label=['多处理器中支持多少个threads是固定的，kernel可以并行和异步']
     kernels-> stream  label=['多个内核异步共同执行']
   }

