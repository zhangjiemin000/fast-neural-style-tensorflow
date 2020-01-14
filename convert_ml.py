# coding: utf-8
import argparse

import tensorflow as tf
from coremltools.models.neural_network import flexible_shape_utils
from tensorflow.core.framework import graph_pb2
import time
import operator
import sys
import os
import tfcoreml

frozen_model_file = os.path.abspath("./transfertransfer.pb")
input_tensor_shapes = {"input_image:0": [1,512,512,3]}
img_input_names = ["input_image:0"]
# Output CoreML model path
coreml_model_file = './model.mlmodel'
output_tensor_names = ['output_image:0']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-print', '--pb_file', help='the path to the pb file')
    parser.add_argument('-out', '--out_text', default='./pbTxt.txt', help='the name of out_txt file')
    return parser.parse_args()



def convert():
    # Read the pb model
    with tf.gfile.GFile(frozen_model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Then, we import the graph_def into a new Graph
    tf.import_graph_def(graph_def, name="")

    # Convert
    tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        # image_input_names=img_input_names,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names)





def inspect(model_pb, output_txt_file):
    graph_def = graph_pb2.GraphDef()
    with open(model_pb, "rb") as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)

    sess = tf.Session()
    OPS = sess.graph.get_operations()

    ops_dict = {}

    sys.stdout = open(output_txt_file, 'w')
    for i, op in enumerate(OPS):
        print('---------------------------------------------------------------------------------------------------------------------------------------------')
        print("{}: op name = {}, op type = ( {} ), inputs = {}, outputs = {}".format(i, op.name, op.type, ", ".join([x.name for x in op.inputs]), ", ".join([x.name for x in op.outputs])))
        print('@input shapes:')
        for x in op.inputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        print('@output shapes:')
        for x in op.outputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        if op.type in ops_dict:
            ops_dict[op.type] += 1
        else:
            ops_dict[op.type] = 1

    print('---------------------------------------------------------------------------------------------------------------------------------------------')
    sorted_ops_count = sorted(ops_dict.items(), key=operator.itemgetter(1))
    print('OPS counts:')
    for i in sorted_ops_count:
        print("{} : {}".format(i[0], i[1]))




if __name__ == '__main__':
    args = parse_args()
    if(args.pb_file is not None):
        inspect(args.pb_file, args.out_text)
    else:
        convert()