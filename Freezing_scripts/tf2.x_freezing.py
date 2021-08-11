# https://stackoverflow.com/questions/58119155/freezing-graph-to-pb-in-tensorflow2

# Method 1
#########################################################################
# I use TF2 to convert model like:

#     pass keras.callbacks.ModelCheckpoint(save_weights_only=True) to model.fit and save checkpoint while training;
#     After training, self.model.load_weights(self.checkpoint_path) load checkpoint, and convert to h5: self.model.save(h5_path, overwrite=True, include_optimizer=False);
#     convert h5 to pb:

import logging
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras

# necessary !!!
tf.compat.v1.disable_eager_execution()

h5_path = '/path/to/model.h5'
model = keras.models.load_model(h5_path)
model.summary()
# save pb
with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, '/path/to/pb/model.pb', as_text=False)
logging.info("save pb successfully！")







# Method 2
############################################################


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import     convert_variables_to_constants_v2
import numpy as np


#set resnet50_v2 as a example
model = tf.keras.applications.ResNet50V2()
 
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
 
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
 
print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
 
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)

# Method 3
######################################################################################3

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib

loaded = tf.saved_model.load('models/mnist_test')
infer = loaded.signatures['serving_default']
f = tf.function(infer).get_concrete_function(
                            flatten_input=tf.TensorSpec(shape=[None, 28, 28, 1], 
                                                        dtype=tf.float32)) # change this line for your own inputs
f2 = convert_variables_to_constants_v2(f)
graph_def = f2.graph.as_graph_def()
if optimize :
    # Remove NoOp nodes
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'NoOp':
            del graph_def.node[i]
    for node in graph_def.node:
        for i in reversed(range(len(node.input))):
            if node.input[i][0] == '^':
                del node.input[i]
    # Parse graph's inputs/outputs
    graph_inputs = [x.name.rsplit(':')[0] for x in frozen_func.inputs]
    graph_outputs = [x.name.rsplit(':')[0] for x in frozen_func.outputs]
    graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def,
                                                                  graph_inputs,
                                                                  graph_outputs,
                                                                  tf.float32.as_datatype_enum)
# Export frozen graph
with tf.io.gfile.GFile('optimized_graph.pb', 'wb') as f:
    f.write(graph_def.SerializeToString())
