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


class LossFunction(ABC):

    def __init__(self, name):

        self.name = name
        self.loss = None

    @abstractmethod
    def build_graph(self):
        pass


class MeanAbsoluteError(LossFunction):

    def __init__(self, prediction, target, weights=1.0, name='Mean_Absolute_Error'):

        LossFunction.__init__(self, name)
        self.prediction = prediction
        self.target = target
        self.weights = weights

    def build_graph(self):

        with tf1.variable_scope('Training/Loss_Functions/' + self.name):
            self.loss = tf1.losses.absolute_difference(labels=self.target,
                                                       predictions=self.prediction,
                                                       weights=tf1.stop_gradient(self.weights),
                                                       scope=None,
                                                       loss_collection=tf1.GraphKeys.LOSSES,
                                                       reduction=tf1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)


class MeanSquaredError(LossFunction):

    def __init__(self, prediction, target, weights=1.0, name='Mean_Squared_Error'):

        LossFunction.__init__(self, name)
        self.prediction = prediction
        self.target = target
        self.weights = weights

    def build_graph(self):

        with tf1.variable_scope('Training/Loss_Functions/' + self.name):
            self.loss = tf1.losses.mean_squared_error(labels=self.target,
                                                      predictions=self.prediction,
                                                      weights=tf1.stop_gradient(self.weights),
                                                      scope=None,
                                                      loss_collection=tf1.GraphKeys.LOSSES,
                                                      reduction=tf1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)