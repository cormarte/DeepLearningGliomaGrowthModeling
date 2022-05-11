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


class Optimizer(ABC):

    def __init__(self, loss, name):

        self.name = name

        self.loss = loss
        self.loss_minimization = None

    @abstractmethod
    def build_graph(self):
        pass


class AdamOptimizer(Optimizer):

    def __init__(self, loss, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam_Optimizer'):

        Optimizer.__init__(self, loss, name)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build_graph(self):

        with tf1.variable_scope('Training/Optimizers/' + self.name):
            optimizer = tf1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=self.beta1,
                                                beta2=self.beta2,
                                                epsilon=self.epsilon,
                                                use_locking=False)

            self.loss_minimization = tf1.group([optimizer.minimize(loss=self.loss, name='Loss_Minimization'),
                                                tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)])  # Required for moving avg update in batch normalization
