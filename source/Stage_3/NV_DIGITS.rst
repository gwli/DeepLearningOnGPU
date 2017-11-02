*************
NVIDIA GIGITS
*************


https://github.com/NVIDIA/DIGITS

#. install CUDA 
#. install nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
#. pip install 

.. code-block:: bash
   
   # example location - can be customized
   DIGITS_ROOT=~/digits
   git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_ROOT
   sudo pip install -r $DIGITS_ROOT/requirements.txt
   sudo pip install -e $DIGITS_ROOT
   ./digits-devserver


https://github.com/NVIDIA/DIGITS/blob/master/docs/BuildDigits.md
