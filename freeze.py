import os
import argparse
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

dir = os.path.dirname(os.path.realpath(__file__))
def freeze_gragh(model_folder):
    ckpt = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = ckpt.model_checkpoint_path

    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"
    print(output_graph)
    output_node_names = "Accuracy/predictions"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')

    input_graph_def = tf.get_default_graph().as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                        output_node_names.split(','))
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
            print(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' %len(output_graph_def.node))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='./results/',help= 'Model folder to export')
    args = parser.parse_args()
    freeze_gragh(args.model_folder)
