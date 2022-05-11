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

import numpy as np
import tensorflow.compat.v1 as tf1
from abc import ABC, abstractmethod
from random import shuffle


class Dataset(ABC):

    def __init__(self, augmenter, generator, max_number_of_cases, batch_size, prefetch_size, shuffle, drop_remainder,
                 name):

        self.augmenter = augmenter
        self.generator = generator
        self.max_number_of_cases = max_number_of_cases
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.name = name

        self.elements = None
        self.number_of_cases = None

        self.initializer = None
        self.next_batch = None

    def build_graph(self):

        indices = list(range(self.number_of_cases))

        if self.max_number_of_cases is not None and self.max_number_of_cases < self.number_of_cases:
            shuffle(indices)
            indices = indices[:self.max_number_of_cases]

        with tf1.variable_scope('Datasets/' + self.name):
            data = tf1.data.Dataset.from_tensor_slices(indices)
            data = data.shuffle(len(indices)) if self.shuffle else data
            data = data.map(lambda index:
                            tuple(tf1.py_func(self.get_augment_and_generate_element,
                                              [index], [tf1.float32 for _ in range(len(self.elements))])),
                            num_parallel_calls=16)
            data = data.batch(self.batch_size, drop_remainder=self.drop_remainder)
            data = data.prefetch(self.prefetch_size)  # Prefetch is in nb of batches since batch is performed before
            iterator = data.make_initializable_iterator()
            self.initializer = iterator.initializer
            self.next_batch = iterator.get_next()

    @abstractmethod
    def get_element(self, index):
        pass

    def get_augment_and_generate_element(self, index):

        element = self.get_element(index)
        element = self.augmenter.augment_element(element) if self.augmenter is not None else element
        element = self.generator.generate_element(element) if self.generator is not None else element

        return element


class ArrayDataset(Dataset):

    def __init__(self, arrays, augmenter=None, generator=None, number_of_cases=None, batch_size=5, prefetch_size=1,
                 shuffle=True, drop_remainder=False, name='Data_Set'):

        Dataset.__init__(self, augmenter, generator, number_of_cases, batch_size, prefetch_size, shuffle, drop_remainder,
                         name)

        self.elements = arrays
        self.number_of_cases = self.elements[0].shape[0]

        assert all([self.elements[i].shape[0] == self.number_of_cases for i in range(len(self.elements))])

    def get_element(self, index):

        return [array[index].astype(np.float32) for array in self.elements]
