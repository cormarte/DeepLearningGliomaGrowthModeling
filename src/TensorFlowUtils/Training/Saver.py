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
from abc import ABC


class Saver(ABC):

    def __init__(self, path, name='Saver'):

        self.path = path
        self.name = name
        self.saver = None

    def build_graph(self):

        with tf1.variable_scope('Saving'):
            self.saver = tf1.train.Saver(name=self.name, max_to_keep=0)

    def save(self, session, epoch, batch, loss):
        self.saver.save(session, self.path + '/' + str(epoch) + '-' + str(batch) + '-' + str(loss), global_step=0)
