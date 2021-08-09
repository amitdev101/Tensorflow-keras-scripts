from typing import List
import os
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def find_meta_info(metafile:str):
    print("### processing file for extracting info '%s' " %(metafile))
    name = metafile.replace('.meta','')
    ckptfile = metafile.replace(".meta",'')
    with tf.Session(graph=tf.Graph()) as sess :
        saver = tf.train.import_meta_graph(metafile)
        saver.restore(sess,ckptfile)
        print("Model Restored Successfully")
        graph_def = tf.get_default_graph().as_graph_def()
        graph_nodes = [node for node in graph_def.node]

        all_nodes = []
        placeholder_nodes = []
        all_nodes_name = []
        all_nodes_name_with_input = []
        possible_output_nodes_name = []

        for node in graph_nodes:
            all_nodes.append(node)
            all_nodes_name.append(node.name)
            all_nodes_name_with_input.extend(node.input)
            if node.op=='Placeholder':
                placeholder_nodes.append(node)
    
        possible_output_nodes_name = list(set(all_nodes_name)-set(all_nodes_name_with_input))
        # placeholder_nodes = [node for node in graph_nodes if node.op=='Placeholder']
        
        def save_info(mylist,filename):
            with open(filename,'w') as f:
                for l in mylist :
                    f.write(str(l)+'\n')
        
        save_info(all_nodes,name+'_all_nodes.txt')
        save_info(all_nodes_name,name+'_all_nodes_name.txt')
        save_info(placeholder_nodes,name+'_place_holder_nodes.txt')
        save_info(possible_output_nodes_name,name+'_possible_output_nodes_name.txt')
        print("### saving info complete ###")
        
            

       

def freeze_pb(metafile:str,output_nodes_list:List[str],name=None): 
    # name is without .pb extension
    print("### Processing meta file for freezing '%s' " %(metafile))
    if not name :
        name = metafile.replace('.meta','')
    pbname = name+'.pb'
    folder_to_save = "./freezed_pb"
    ckptfile = metafile.replace(".meta",'')
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(metafile)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver.restore(sess,ckptfile)
        print("Model Restored Successfully")
        output_nodes=output_nodes_list
        frozen_graph =  freeze_session (sess, output_names=output_nodes) # provide your output node list: by default freezing all nodes names
        tf.train.write_graph(frozen_graph, folder_to_save, pbname, as_text=False) 
        print("Saved file in frozen_pb format in folder %s named as '%s'" %(folder_to_save,pbname))


if __name__=="__main__":
    metafiles = [f.name for f in os.scandir() if not f.is_dir() and (f.name).endswith('.meta')]
    for metafile in metafiles :
        # find_meta_info(metafile)
        pass
    ### freezing pb ###
    output_nodes_list = ['Placeholder','generator/Conv_9/BiasAdd']
    freeze_pb(metafiles[0],output_nodes_list,name='cartoon-conv9')
    