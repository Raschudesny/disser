#import tensorflow as tf
import numpy as np

"""
def make2DconvolutionWith3Dfilter(images, filter):
    sess = tf.Session()
    #ones_3d = np.ones((5, 5, 5))
    ones_3d = images
    print('ones_3d = ', ones_3d)
    #weight_3d = np.ones((3, 3, 3))
    weight_3d = filter
    print('weight_3d = ', weight_3d)
    strides_3d = [1, 1, 1, 1, 1]

    in_3d = tf.constant(ones_3d, dtype=tf.float32)
    print('in_3d = \n', in_3d)
    filter_3d = tf.constant(weight_3d, dtype=tf.float32)
    print('filter_3d = \n', filter_3d)

    in_width = int(in_3d.shape[2])
    in_height = int(in_3d.shape[1])
    in_depth = int(in_3d.shape[0])

    filter_width = int(filter_3d.shape[2])
    filter_height = int(filter_3d.shape[1])
    filter_depth = int(filter_3d.shape[0])

    input_3d = tf.reshape(in_3d, [1, in_depth, in_height, in_width, 1])
    print('input_3d = \n', input_3d)
    kernel_3d = tf.reshape(filter_3d, [filter_depth, filter_height, filter_width, 1, 1])
    print('kernel_3d = \n', kernel_3d)

    output_3d = tf.squeeze(tf.nn.conv3d(input_3d, kernel_3d, strides=strides_3d, padding='SAME'))
    result = sess.run(output_3d)
    print('RESULT = \n', result)
    return result
"""