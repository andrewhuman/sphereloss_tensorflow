import os
from tensorflow.python import pywrap_tensorflow
checkpoint_path = 'modeltrained/20170512/model-20170512-110547.ckpt-250000'

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
print(len(var_to_shape_map))
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(reader.get_tensor(key).shape)
