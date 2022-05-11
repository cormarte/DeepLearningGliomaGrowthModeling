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
from TensorFlowUtils.Networks.Network import Network


class ConvolutionalRegressor(Network):

    def __init__(self, nb_of_convolutional_features, kernel_sizes, nb_of_fully_connected_features, 
                 down_sampling_factor=2, down_sampling_mode='convolution', activation=tf1.nn.relu, batch_norm=True,
                 output_bias=True, output_activation=None, global_avg=True, name='ConvolutionalRegressor'):

        Network.__init__(self, nb_of_fully_connected_features[-1], name)

        self.nb_of_convolutional_features = nb_of_convolutional_features
        self.kernel_sizes = kernel_sizes
        self.nb_of_fully_connected_features = nb_of_fully_connected_features
        self.down_sampling_factor = down_sampling_factor
        self.down_sampling_mode = down_sampling_mode
        self.activation = activation
        self.batch_norm = batch_norm
        self.output_bias = output_bias
        self.output_activation = output_activation
        self.global_avg = global_avg

        self.fully_connected_input = None

    def build_convolutional_block(self, input, kernel_sizes, nb_of_convolutional_features):

        flow = input

        for i in range(len(kernel_sizes)):
            with tf1.variable_scope('Convolution_Layer_' + str(i)):
                flow = self.convolution(input=flow,
                                        kernel_size=kernel_sizes[i],
                                        nb_of_features=nb_of_convolutional_features[i],
                                        strides=1,
                                        padding='same',
                                        activation=self.activation,
                                        normalization=tf1.layers.batch_normalization if self.batch_norm else None,
                                        bias_initializer=None if self.batch_norm else tf1.constant_initializer(0.1))

        with tf1.variable_scope('Downsampling_Layer'):
            if self.down_sampling_mode == 'convolution':
                flow = self.convolution(input=flow,
                                        kernel_size=self.down_sampling_factor,
                                        nb_of_features=nb_of_convolutional_features[-1],
                                        strides=self.down_sampling_factor,
                                        padding='valid',
                                        activation=self.activation,
                                        normalization=tf1.layers.batch_normalization if self.batch_norm else None,
                                        bias_initializer=None if self.batch_norm else tf1.constant_initializer(0.1))

            elif self.down_sampling_mode == 'mean_pool':
                flow = self.mean_pooling(input=flow,
                                         kernel_size=self.down_sampling_factor,
                                         strides=self.down_sampling_factor,
                                         padding='valid')

            else:
                flow = self.max_pooling(input=flow,
                                        kernel_size=self.down_sampling_factor,
                                        strides=self.down_sampling_factor,
                                        padding='valid')

        return flow

    def build_fully_connected_block(self, input, nb_of_fully_connected_features):

        with tf1.variable_scope('Fully_Connected_Layer'):
            flow = self.fully_connected(input=input,
                                        nb_of_units=nb_of_fully_connected_features,
                                        activation=self.activation,
                                        normalization=tf1.layers.batch_normalization if self.batch_norm else None,
                                        bias_initializer=None if self.batch_norm else tf1.constant_initializer(0.1))

        return flow

    def build_graph(self):

        with tf1.variable_scope('Network/' + self.name):

            # Convolutional part
            flow = self.input
            for i in range(len(self.kernel_sizes)):
                with tf1.variable_scope('Convolutional_Block_' + str(i + 1)):
                    flow = self.build_convolutional_block(flow, self.kernel_sizes[i], self.nb_of_convolutional_features[i])

            # Fully connected part
            if self.global_avg:
                flow = self.global_average(flow)

            flow = self.flatten(flow)

            if self.fully_connected_input is not None:
                flow = self.concat(input1=flow,
                                   input2=self.fully_connected_input,
                                   axis=-1)

            for i in range(len(self.nb_of_fully_connected_features)-1):
                with tf1.variable_scope('Fully_Connected_Block_' + str(i + 1)):
                    flow = self.build_fully_connected_block(flow, self.nb_of_fully_connected_features[i])

            # Output part
            with tf1.variable_scope('Output_Block'):
                self.output = self.fully_connected(input=flow,
                                                   nb_of_units=self.nb_of_fully_connected_features[-1],
                                                   activation=self.output_activation,
                                                   normalization=None,
                                                   bias_initializer=tf1.constant_initializer(0.1) if self.output_bias else None)
