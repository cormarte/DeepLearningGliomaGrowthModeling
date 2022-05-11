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
from datetime import datetime
from TensorFlowUtils.Data.Datasets import ArrayDataset
from TensorFlowUtils.Data.Input import Input
from TensorFlowUtils.Networks.UNet import UNet
from TensorFlowUtils.Training.LossFunctions import MeanAbsoluteError
from TensorFlowUtils.Training.Optimizers import AdamOptimizer
from TensorFlowUtils.Training.Saver import Saver
from os import remove
from os.path import join
from random import randint
from sortedcontainers import SortedDict


def train():

    # Trains a 3D U-Net for the estimation of the tumor cell density distribution from two threshold-like imaging
    # contours (from a single imaging time point) and a diffusion tensor field

    tf1.disable_v2_behavior()  # This code was written for TF v1 but TF v2 is required for RTX 30xx GPUs

    epochs = 5000   # The maximum number of epochs
    augment = True  # Whether to perform augmentation

    directory = 'D:/Datasets/Tumour Growth/distribution'  # Base dataset directory, to be adapted

    # Array loading, to be adapted
    # For memory efficiency, diffusion tensors are stored only once for the 200 synthetic tumors of each subject
    # training_case_array and testing_case_array hold the case ids of each tumor to retrieve the corresponding tensor
    training_source_array = np.load(join(directory, 'Training/Source.npy'))
    training_target_array = np.load(join(directory, 'Training/Target.npy'))
    training_case_array = np.load(join(directory, 'Training/Case.npy'))
    training_tensor_array = np.load(join(directory, 'Training/Tensor.npy'))
    testing_source_array = np.load(join(directory, 'Testing/Source.npy'))
    testing_target_array = np.load(join(directory, 'Testing/Target.npy'))
    testing_case_array = np.load(join(directory, 'Testing/Case.npy'))
    testing_tensor_array = np.load(join(directory, 'Testing/Tensor.npy'))

    training_dataset = ArrayDataset(arrays=[training_source_array, training_target_array, training_case_array],
                                    augmenter=None,
                                    generator=None,
                                    number_of_cases=None,
                                    batch_size=1,
                                    prefetch_size=1,
                                    shuffle=True,
                                    drop_remainder=True,
                                    name='Training_Data_Set')
    training_dataset.build_graph()

    testing_dataset = ArrayDataset(arrays=[testing_source_array, testing_target_array, testing_case_array],
                                   augmenter=None,
                                   generator=None,
                                   number_of_cases=None,
                                   batch_size=1,
                                   prefetch_size=1,
                                   shuffle=False,
                                   drop_remainder=False,
                                   name='Testing_Data_Set')
    testing_dataset.build_graph()

    # A simple placeholder for the source imaging contours and diffusion tensor
    source = Input(dimensions=[None, 192, 192, 128, 8], name="Source")
    source.build_graph()

    # A simple placeholder for the training flag, used for batch norm if enabled
    training_flag = Input(dimensions=None, name="Training_Flag", dtype=tf1.bool)
    training_flag.build_graph()

    network = UNet(nb_of_features=[[32, 32, 32], [32, 32, 64], [64, 64, 64], [64, 64, 128], [128, 128, 64], [64, 64, 64],
                                   [64, 64, 32], [32, 32, 32], [32, 32, 1]],
                   kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                   down_sampling_factor=2,
                   down_sampling_mode='convolution',
                   activation=tf1.nn.relu,
                   batch_norm=False,
                   output_bias=True,
                   output_activation=None,
                   name='UNet3D')
    network.input = source.data
    network.training_flag = training_flag.data
    network.build_graph()

    # A simple placeholder for the target tumor cell distribution
    target = Input(dimensions=[None] + network.output.get_shape().as_list()[1:-1] + [1], name="Target")
    target.build_graph()

    loss_function = MeanAbsoluteError(prediction=network.output,
                                      target=target.data,
                                      weights=1.0,
                                      name='Mean_Absolute_Error')
    loss_function.build_graph()

    optimizer = AdamOptimizer(loss_function.loss,
                              learning_rate=1e-4,
                              beta1=0.9,
                              beta2=0.999,
                              epsilon=1e-06)
    optimizer.build_graph()

    saver = Saver('./Networks/' + network.name + '/Current')  # Output network directory, to be adapted
    saver.build_graph()

    # To keep a track of the 5 best networks in terms of testing loss
    best_testing_losses = SortedDict({})
    no_improvement_counter = 0

    config = tf1.ConfigProto()

    with tf1.Session(config=config) as session:

        writer = tf1.summary.FileWriter('./Networks/' + network.name + './Current', session.graph)
        session.run(tf1.global_variables_initializer())

        for e in range(epochs):

            b = 0
            session.run(training_dataset.initializer)

            while True:
                try:
                    b += 1

                    # training_case is the subject index allowing to retrieve the corresponding diffusion tensor
                    # (stored only once for the 200 tumors of each subject)
                    training_source, training_target, training_case = session.run(training_dataset.next_batch)
                    training_source = np.concatenate((training_source, training_tensor_array[
                        np.squeeze(training_case, axis=-1).astype(np.uint8)]), axis=-1)

                    # Simple shift augmentation
                    if augment:

                        shift = tuple([randint(-15, 15) for _ in range(3)])
                        training_source = np.roll(training_source, shift=shift, axis=(1, 2, 3))
                        training_target = np.roll(training_target, shift=shift, axis=(1, 2, 3))

                        if shift[0] >= 0:
                            training_source[:, :shift[0], :, :, :] = 0.0
                            training_target[:, :shift[0], :, :, :] = 0.0
                        else:
                            training_source[:, shift[0]:, :, :, :] = 0.0
                            training_target[:, shift[0]:, :, :, :] = 0.0

                        if shift[1] >= 0:
                            training_source[:, :, :shift[1], :, :] = 0.0
                            training_target[:, :, :shift[1], :, :] = 0.0
                        else:
                            training_source[:, :, shift[1]:, :, :] = 0.0
                            training_target[:, :, shift[1]:, :, :] = 0.0

                        if shift[2] >= 0:
                            training_source[:, :, :, :shift[2], :] = 0.0
                            training_target[:, :, :, :shift[2], :] = 0.0
                        else:
                            training_source[:, :, :, shift[2]:, :] = 0.0
                            training_target[:, :, :, shift[2]:, :] = 0.0

                    session.run(optimizer.loss_minimization, feed_dict={source.data: training_source,
                                                                        target.data: training_target,
                                                                        training_flag.data: True})

                except tf1.errors.OutOfRangeError:  # End of dataset, performs evaluation

                    # Testing loss
                    testing_sum = 0.0
                    testing_n = 0.0

                    session.run(testing_dataset.initializer)

                    while True:
                        try:
                            # testing_case is the subject index allowing to retrieve the corresponding diffusion tensor
                            # (stored only once for the 200 tumors of each subject)
                            testing_source, testing_target, testing_case = session.run(testing_dataset.next_batch)
                            testing_source = np.concatenate((testing_source, testing_tensor_array[
                                np.squeeze(testing_case, axis=-1).astype(np.uint8)]), axis=-1)
                            testing_sum += session.run(loss_function.loss, feed_dict={source.data: testing_source,
                                                                                      target.data: testing_target,
                                                                                      training_flag.data: False})
                            testing_n += 1.0
                        except tf1.errors.OutOfRangeError:
                            break

                    testing_loss = testing_sum / testing_n

                    # Training loss
                    training_sum = 0.0
                    training_n = 0.0

                    session.run(training_dataset.initializer)

                    while True:
                        try:
                            # training_case is the subject index allowing to retrieve the corresponding diffusion tensor
                            # (stored only once for the 200 tumors of each subject)
                            training_source, training_target, training_case = session.run(training_dataset.next_batch)
                            training_source = np.concatenate((training_source, training_tensor_array[
                                np.squeeze(training_case, axis=-1).astype(np.uint8)]), axis=-1)
                            training_sum += session.run(loss_function.loss, feed_dict={source.data: training_source,
                                                                                       target.data: training_target,
                                                                                       training_flag.data: False})
                            training_n += 1.0
                        except tf1.errors.OutOfRangeError:
                            break

                    training_loss = training_sum/training_n

                    # Keeps the 5 best networks in terms of testing loss
                    if len(best_testing_losses) < 5 or testing_loss < best_testing_losses.keys()[-1]:

                        file = str(e + 1) + "-" + str(b) + "-" + str(testing_loss) + "-0"

                        if len(best_testing_losses) == 0 or testing_loss < best_testing_losses.keys()[0]:
                            no_improvement_counter = 0
                        else:
                            no_improvement_counter += 1

                        if len(best_testing_losses) >= 5:
                            remove(saver.path + '/' + best_testing_losses.values()[-1] + '.data-00000-of-00001')
                            remove(saver.path + '/' + best_testing_losses.values()[-1] + '.index')
                            remove(saver.path + '/' + best_testing_losses.values()[-1] + '.meta')
                            best_testing_losses.popitem(-1)

                        saver.save(session, e + 1, b, testing_loss)
                        best_testing_losses[testing_loss] = file
                        print(best_testing_losses)

                    else:
                        no_improvement_counter += 1

                    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch {e + 1}: Training loss: {training_loss}, testing loss: {testing_loss}, best testing loss: {best_testing_losses.keys()[0]}, no improvement in {no_improvement_counter} epochs.')
                    break

        writer.close()


def predict():

    tf1.disable_v2_behavior()  # This code was written for TF v1 but TF v2 is required for RTX 30xx GPUs

    # Trained network directory and name
    network_directory = ''
    network_name = ''

    base_directory = ''  # Base dataset directory, to be adapted
    source_array = np.load(join(base_directory, 'Source.npy'))
    case_array = np.load(join(base_directory, 'Case.npy'))
    tensor_array = np.load(join(base_directory, 'Tensor.npy'))
    prediction_file = join(base_directory, 'Prediction.npy')
    prediction_array = np.zeros(source_array.shape[:-1] + (1,), dtype=np.float32)

    dataset = ArrayDataset(arrays=[source_array, case_array],
                           augmenter=None,
                           generator=None,
                           number_of_cases=None,
                           batch_size=1,
                           prefetch_size=1,
                           shuffle=False,
                           drop_remainder=False,
                           name='Data_Set')
    dataset.build_graph()

    config = tf1.ConfigProto()

    with tf1.Session(config=config) as session:

        loader = tf1.train.import_meta_graph(network_directory + network_name + '.meta')
        loader.restore(session, network_directory + network_name)
        graph = tf1.get_default_graph()

        source = graph.get_tensor_by_name('Inputs/Source:0')
        training_flag = graph.get_tensor_by_name('Inputs/Training_Flag:0')
        prediction = graph.get_tensor_by_name('Network/UNet3D/Output_Block/Output_Layer/Convolution/conv3d/BiasAdd:0')

        session.run(dataset.initializer)
        index = 0

        while True:
            try:
                batch_source, batch_case = session.run(dataset.next_batch)
                batch_source = np.concatenate((batch_source, tensor_array[np.squeeze(batch_case, axis=-1).astype(np.uint8)]), axis=-1)
                batch_prediction = session.run(prediction, feed_dict={source: batch_source, training_flag: False})
                prediction_array[index] = batch_prediction[0]
                index += 1

            except tf1.errors.OutOfRangeError:
                break

    np.save(prediction_file, prediction_array)


if __name__ == '__main__':

    train()
    # predict()
