# Copyright (C) 2022  Corentin Martens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# If you use this software for your research, please cite "Martens et al.
# Deep Learning for Reaction-Diffusion Glioma Growth Modelling: Towards a
# Fully Personalised Model? arXiv:2111.13404, Nov 2021."
#
# Contact: corentin.martens@ulb.be || corentin.martens@hotmail.be

import tensorflow.compat.v1 as tf1
from abc import ABC, abstractmethod


class Network(ABC):

    def __init__(self, nb_of_output_channels, name):

        self.nb_of_output_channels = nb_of_output_channels
        self.name = name

        self.input = None
        self.training_flag = None
        self.output = None

    def convolution(self, input, kernel_size, nb_of_features, strides=1, padding='valid', activation=tf1.nn.relu,
                    normalization=tf1.layers.batch_normalization, bias_initializer=tf1.constant_initializer(0.1)):

        rank = input.get_shape().ndims-2
        func = {1: tf1.layers.conv1d, 2: tf1.layers.conv2d, 3: tf1.layers.conv3d}[rank]

        with tf1.variable_scope('Convolution'):

            flow = func(inputs=input,
                        filters=nb_of_features,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format='channels_last',
                        dilation_rate=1,
                        activation=None,
                        use_bias=normalization is None and bias_initializer is not None,
                        kernel_initializer=tf1.variance_scaling_initializer(scale=2.0, mode='fan_avg'),
                        bias_initializer=bias_initializer,
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name=None,
                        reuse=None)

            if normalization is not None:
                flow = normalization(flow, training=self.training_flag)

            if activation is not None:
                flow = activation(flow)

        return flow

    @staticmethod
    def concat(input1, input2, axis=-1):

        with tf1.variable_scope('Concatenation'):

            flow = tf1.concat(values=[input1, input2], axis=axis, name=None)

        return flow

    @staticmethod
    def crop(input, output):

        with tf1.variable_scope('Cropping'):

            input_dimensions = input.shape.as_list()
            output_dimensions = output.shape.as_list()
            begin = tf1.convert_to_tensor([0] + [(input_dimensions[i]-output_dimensions[i])//2
                                                 for i in range(1, len(input_dimensions)-1)] + [0], dtype=tf1.int32)
            size = tf1.convert_to_tensor([-1] + output_dimensions[1:-1] + [input_dimensions[-1]], dtype=tf1.int32)

            flow = tf1.slice(input_=input, begin=begin, size=size, name=None)

        return flow

    def deconvolution(self, input, kernel_size, nb_of_features, strides=1, padding='valid', activation=tf1.nn.relu,
                      normalization=tf1.layers.batch_normalization, bias_initializer=tf1.constant_initializer(0.1)):

        rank = input.get_shape().ndims-2
        func = {1: tf1.layers.conv2d_transpose, 2: tf1.layers.conv2d_transpose, 3: tf1.layers.conv3d_transpose}[rank]

        with tf1.variable_scope('Deconvolution'):

            if rank == 1:  # tf1.layers.conv1d_transpose does not exist
                input = tf1.reshape(input, [-1, 1] + input.get_shape().as_list()[1:])
                kernel_size = [1, kernel_size]
                strides = [1, strides]

            flow = func(inputs=input,
                        filters=nb_of_features,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format='channels_last',
                        activation=None,
                        use_bias=normalization is None and bias_initializer is not None,
                        kernel_initializer=tf1.variance_scaling_initializer(scale=2.0, mode='fan_avg'),
                        bias_initializer=bias_initializer,
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name=None,
                        reuse=None)

            if normalization is not None:
                flow = normalization(flow, training=self.training_flag)

            if activation is not None:
                flow = activation(flow)

            if rank == 1:
                flow = tf1.reshape(flow, [-1] + flow.get_shape().as_list()[2:])

        return flow

    @staticmethod
    def global_average(input):

        rank = input.get_shape().ndims

        with tf1.variable_scope('Global_Average'):

            flow = tf1.reduce_mean(input_tensor=input,
                                   axis=list(range(1, rank-1)),
                                   keepdims=None,
                                   name=None,
                                   reduction_indices=None,
                                   keep_dims=None)

        return flow

    def fully_connected(self, input, nb_of_units, activation=tf1.nn.relu,
                        normalization=tf1.layers.batch_normalization,
                        bias_initializer=tf1.constant_initializer(0.1)):

        with tf1.variable_scope('Fully_Connected'):

            flow = tf1.layers.dense(inputs=input,
                                    units=nb_of_units,
                                    activation=activation,
                                    use_bias=normalization is None and bias_initializer is not None,
                                    kernel_initializer=tf1.glorot_uniform_initializer(),
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name=None,
                                    reuse=None)

            if normalization is not None:
                flow = normalization(flow, training=self.training_flag)

            if activation is not None:
                flow = activation(flow)

        return flow

    @staticmethod
    def flatten(input):

        with tf1.variable_scope('Flattening'):

            flow = tf1.layers.flatten(inputs=input, name=None, data_format='channels_last')

        return flow

    @staticmethod
    def max_pooling(input, kernel_size, strides=1, padding='valid'):

        rank = input.get_shape().ndims-2
        func = {1: tf1.layers.max_pooling1d, 2: tf1.layers.max_pooling2d, 3: tf1.layers.max_pooling3d}[rank]

        with tf1.variable_scope('Max_Pooling'):

            flow = func(intputs=input,
                        pool_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format='channels_last',
                        name=None)

        return flow

    @staticmethod
    def mean_pooling(input, kernel_size, strides=1, padding='valid'):

        rank = input.get_shape().ndims - 2
        func = {1: tf1.layers.average_pooling1d, 2: tf1.layers.average_pooling2d, 3: tf1.layers.average_pooling3d}[rank]

        with tf1.variable_scope('Mean_Pooling'):

            flow = func(inputs=input,
                        pool_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format='channels_last',
                        name=None)

        return flow

    @abstractmethod
    def build_graph(self):
        pass
