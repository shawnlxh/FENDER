# KDD2022 submission  

The source code of FENDER.  
The model is developed by Tensorflow 1.12. Implementation collaborated with Zheng Liu.    

Data:  
instacart:  
https://www.instacart.com/datasets/grocery-shopping-2017
May need to require access from administrator.

preprocess data: instacart_pre.ipynb, you may see the preprocessing steps in the file.  
The processed files are too large to push it on github, but you can get the correct data format with the preprocessing code.  

For the tensorflow version > 2.0, please do:  
replace  
import tensorflow as tf  
with  
import tensorflow.compat.v1 as tf  
tf.disable_v2_behavior()  

The factorization of PIF is from NCF  
https://github.com/hexiangnan/neural_collaborative_filtering  
Another simpler way to factorize PIF, run the following code:  
python mf.py --dataset instacart

To train the model and test, go the Model folder and run the following command:  
python Main.py --model_type fender --dataset instacart --gpu_id 0

