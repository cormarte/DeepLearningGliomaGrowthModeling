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


class UNet(Network):

    def __init__(self, nb_of_features, kernel_sizes, down_sampling_factor=2, down_sampling_mode='convolution',
                 activation=tf1.nn.relu, batch_norm=True, output_bias=True, output_activation=None,
                 name='UNet'):

        Network.__init__(self, nb_of_features[-1], name)

        self.nb_of_features = nb_of_features
        self.kernel_sizes = kernel_sizes
        self.down_sampling_factor = down_sampling_factor
        self.down_sampling_mode = down_sampling_mode
        self.activation = activation
        self.batch_norm = batch_norm
        self.output_bias = output_bias
        self.output_activation = output_activation

        self.depth = len(nb_of_features)//2

    def build_block(self, input, kernel_sizes, nb_of_features, function):

        flow = input

        for i in range(len(kernel_sizes)):
            with tf1.variable_scope('Convolution_Layer_' + str(i)):
                flow = self.convolution(input=flow,
                                        kernel_size=kernel_sizes[i],
                                        nb_of_features=nb_of_features[i],
                                        strides=1,
                                        padding='same',
                                        activation=self.activation,
                                        normalization=tf1.layers.batch_normalization if self.batch_norm else None,
                                        bias_initializer=None if self.batch_norm else tf1.constant_initializer(0.1))
        bypass = flow

        if function == 'Down':
            with tf1.variable_scope('Downsampling_Layer'):
                if self.down_sampling_mode == 'convolution':
                    flow = self.convolution(input=flow,
                                            kernel_size=self.down_sampling_factor,
                                            nb_of_features=nb_of_features[-1],
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

        elif function == 'Up':
            with tf1.variable_scope('Upsampling_Layer'):
                flow = self.deconvolution(input=flow,
                                          kernel_size=self.down_sampling_factor,
                                          nb_of_features=nb_of_features[-1],
                                          strides=self.down_sampling_factor,
                                          padding='valid',
                                          activation=self.activation,
                                          normalization=tf1.layers.batch_normalization if self.batch_norm else None,
                                          bias_initializer=None if self.batch_norm else tf1.constant_initializer(0.1))

        elif function == 'Output':
            with tf1.variable_scope('Output_Layer'):
                flow = self.convolution(input=flow,
                                        kernel_size=1,
                                        nb_of_features=nb_of_features[-1],
                                        strides=1,
                                        padding='same',
                                        activation=self.output_activation,
                                        normalization=None,
                                        bias_initializer=tf1.constant_initializer(0.1) if self.output_bias else None)

        return flow, bypass

    def build_graph(self):

        with tf1.variable_scope('Network/' + self.name):

            flow = self.input
            block_id = 0

            # Contracting part
            bypasses = []
            for i in range(self.depth):
                with tf1.variable_scope('Downsampling_Block_' + str(i+1)):
                    flow, bypass = self.build_block(flow, self.kernel_sizes[block_id], self.nb_of_features[block_id], 'Down')
                    bypasses.append(bypass)
                    block_id += 1

            # Expanding part
            for i in range(self.depth):
                with tf1.variable_scope('Upsampling_Block_' + str(i+1)):
                    if i != 0:
                        bypass_cropped = self.crop(bypasses[-i], flow)
                        flow = self.concat(flow, bypass_cropped, -1)
                    flow, _ = self.build_block(flow, self.kernel_sizes[block_id], self.nb_of_features[block_id], 'Up')
                    block_id += 1

            # Output
            with tf1.variable_scope('Output_Block'):
                bypass_cropped = self.crop(bypasses[0], flow)
                flow = self.concat(flow, bypass_cropped, -1)
                flow, _ = self.build_block(flow, self.kernel_sizes[-1], self.nb_of_features[-1], 'Output')

            self.output = flow
