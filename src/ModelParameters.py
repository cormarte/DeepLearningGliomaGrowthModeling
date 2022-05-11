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
from TensorFlowUtils.Networks.ConvolutionalRegressor import ConvolutionalRegressor
from TensorFlowUtils.Training.LossFunctions import MeanSquaredError
from TensorFlowUtils.Training.Optimizers import AdamOptimizer
from TensorFlowUtils.Training.Saver import Saver
from os import remove
from os.path import join
from random import randint
from sortedcontainers import SortedDict


def train():

    # Trains a 3D convolutional encoder for the estimation of the model parameters from three threshold-like imaging
    # contours (from two imaging time points), a diffusion tensor field, and the interval between the imaging times

    tf1.disable_v2_behavior()  # This code was written for TF v1 but TF v2 is required for RTX 30xx GPUs

    epochs = 5000   # The maximum number of epochs
    augment = True  # Whether to perform augmentation

    directory = ''  # Base dataset directory, to be adapted

    # Array loading, to be adapted
    # For memory efficiency, diffusion tensors are stored only once for the 200 synthetic tumors of each subject
    # training_case_array and testing_case_array hold the case ids of each tumor to retrieve the corresponding tensor
    training_source_array_1 = np.load(join(directory, 'Training/Source_1.npy'))  # Imaging contours
    training_source_array_2 = np.load(join(directory, 'Training/Source_2.npy'))  # Time intervals
    training_target_array = np.load(join(directory, 'Training/Target.npy'))
    training_case_array = np.load(join(directory, 'Training/Case.npy'))
    training_tensor_array = np.load(join(directory, 'Training/Tensor.npy'))
    testing_source_array_1 = np.load(join(directory, 'Testing/Source_1.npy'))
    testing_source_array_2 = np.load(join(directory, 'Testing/Source_2.npy'))
    testing_target_array = np.load(join(directory, 'Testing/Target.npy'))
    testing_case_array = np.load(join(directory, 'Testing/Case.npy'))
    testing_tensor_array = np.load(join(directory, 'Testing/Tensor.npy'))

    # Standardization of the parameters
    l_min = np.sqrt(10.0/40.0)
    l_max = np.sqrt(40.0/10.0)
    l_mean = 0.5*(l_min+l_max)           # Theoretical mean of a uniform distribution
    l_std = (l_max-l_min)/np.sqrt(12.0)  # Theoretical std of a uniform distribution
    v_min = 2.0*np.sqrt(200.0)
    v_max = 2.0*np.sqrt(600.0)
    v_mean = 0.5*(v_min+v_max)           # Theoretical mean of a uniform distribution
    v_std = (v_max-v_min)/np.sqrt(12.0)  # Theoretical std of a uniform distribution
    mean = np.array([l_mean, v_mean], dtype=np.float32)
    std = np.array([l_std, v_std], dtype=np.float32)
    training_target_array = (training_target_array-mean)/std
    testing_target_array = (testing_target_array-mean)/std

    training_dataset = ArrayDataset(arrays=[training_source_array_1, training_source_array_2, training_target_array, training_case_array],
                                    augmenter=None,
                                    generator=None,
                                    number_of_cases=None,
                                    batch_size=1,
                                    prefetch_size=1,
                                    shuffle=True,
                                    drop_remainder=True,
                                    name='Training_Data_Set')
    training_dataset.build_graph()

    testing_dataset = ArrayDataset(arrays=[testing_source_array_1, testing_source_array_2, testing_target_array, testing_case_array],
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
    source_1 = Input(dimensions=[None, 192, 192, 128, 9], name="Source_1")
    source_1.build_graph()

    # A simple placeholder for the source time interval (scalar)
    source_2 = Input(dimensions=[None, 1], name="Source_2")
    source_2.build_graph()

    # A simple placeholder for the training flag, used for batch norm if enabled
    training_flag = Input(dimensions=None, name="Training_Flag", dtype=tf1.bool)
    training_flag.build_graph()

    network = ConvolutionalRegressor(nb_of_convolutional_features=[[32, 32, 64], [64, 64, 128], [128, 128, 64],
                                                                   [64, 64, 32], [32, 32, 16], [16, 16, 8]],
                                     kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                                     nb_of_fully_connected_features=[2],
                                     down_sampling_factor=2,
                                     down_sampling_mode='convolution',
                                     activation=tf1.nn.relu,
                                     batch_norm=False,
                                     output_bias=True,
                                     output_activation=None,
                                     global_avg=False,
                                     name='ConvolutionnalRegressor3D')
    network.input = source_1.data
    network.fully_connected_input = source_2.data
    network.training_flag = training_flag.data
    network.build_graph()

    # A simple placeholder for the target parameter values (scalars)
    target = Input(dimensions=[None] + network.output.get_shape().as_list()[1:],
                   name="Target")
    target.build_graph()

    loss_function = MeanSquaredError(prediction=network.output,
                                     target=target.data,
                                     name='Mean_Squared_Error')
    loss_function.build_graph()

    optimizer = AdamOptimizer(loss_function.loss,
                              learning_rate=1e-5,
                              beta1=0.9,
                              beta2=0.999,
                              epsilon=1e-06)
    optimizer.build_graph()

    saver = Saver('./Networks/' + network.name + '/Current')
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
                    training_source_1, training_source_2, training_target, training_case = session.run(training_dataset.next_batch)
                    training_source_1 = np.concatenate((training_source_1, training_tensor_array[
                        np.squeeze(training_case, axis=-1).astype(np.uint8)]), axis=-1)

                    # Simple shift augmentation
                    if augment:

                        shift = tuple([randint(-15, 15) for _ in range(3)])
                        training_source_1 = np.roll(training_source_1, shift=shift, axis=(1, 2, 3))

                        if shift[0] >= 0:
                            training_source_1[:, :shift[0], :, :, :] = 0.0
                        else:
                            training_source_1[:, shift[0]:, :, :, :] = 0.0

                        if shift[1] >= 0:
                            training_source_1[:, :, :shift[1], :, :] = 0.0
                        else:
                            training_source_1[:, :, shift[1]:, :, :] = 0.0

                        if shift[2] >= 0:
                            training_source_1[:, :, :, :shift[2], :] = 0.0
                        else:
                            training_source_1[:, :, :, shift[2]:, :] = 0.0

                    session.run(optimizer.loss_minimization, feed_dict={source_1.data: training_source_1,
                                                                        source_2.data: training_source_2,
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
                            testing_source_1, testing_source_2, testing_target, testing_case = session.run(testing_dataset.next_batch)
                            testing_source_1 = np.concatenate((testing_source_1, testing_tensor_array[
                                np.squeeze(testing_case, axis=-1).astype(np.uint8)]), axis=-1)
                            testing_sum += session.run(loss_function.loss, feed_dict={source_1.data: testing_source_1,
                                                                                      source_2.data: testing_source_2,
                                                                                      target.data: testing_target,
                                                                                      training_flag.data: False})
                            testing_n += 1.0
                        except tf1.errors.OutOfRangeError:
                            break

                    testing_loss = testing_sum/testing_n

                    # Training loss
                    training_sum = 0.0
                    training_n = 0.0

                    session.run(training_dataset.initializer)

                    while True:
                        try:
                            # training_case is the subject index allowing to retrieve the corresponding diffusion tensor
                            # (stored only once for the 200 tumors of each subject)
                            training_source_1, training_source_2, training_target, training_case = session.run(training_dataset.next_batch)
                            training_source_1 = np.concatenate((training_source_1, training_tensor_array[
                                np.squeeze(training_case, axis=-1).astype(np.uint8)]), axis=-1)
                            training_sum += session.run(loss_function.loss, feed_dict={source_1.data: training_source_1,
                                                                                       source_2.data: training_source_2,
                                                                                       target.data: training_target,
                                                                                       training_flag.data: False})
                            training_n += 1.0
                        except tf1.errors.OutOfRangeError:
                            break

                    training_loss = training_sum/training_n

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

                        saver.save(session, e+1, b, testing_loss)
                        best_testing_losses[testing_loss] = file
                        print(best_testing_losses)

                    else:
                        no_improvement_counter += 1

                    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch {e+1}: Training loss: {training_loss}, testing loss: {testing_loss}, best testing loss: {best_testing_losses.keys()[0]}, no improvement in {no_improvement_counter} epochs.')
                    break

        writer.close()


def predict():

    tf1.disable_v2_behavior()  # This code was written for TF v1 but TF v2 is required for RTX 30xx GPUs

    # Trained network directory and name
    network_directory = ''
    network_name = ''

    base_directory = ''  # Base dataset directory, to be adapted
    source_array_1 = np.load(join(base_directory, 'Source_1.npy'))
    source_array_2 = np.load(join(base_directory, 'Source_2.npy'))
    case_array = np.load(join(base_directory, 'Case.npy'))
    tensor_array = np.load(join(base_directory, 'Tensor.npy'))
    target_array = np.load(join(base_directory, 'Target.npy'))
    prediction_file = join(base_directory, 'Prediction.npy')
    prediction_array = np.zeros_like(target_array)

    dataset = ArrayDataset(arrays=[source_array_1, source_array_2, case_array],
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

        source_1 = graph.get_tensor_by_name('Inputs/Source_1:0')
        source_2 = graph.get_tensor_by_name('Inputs/Source_2:0')
        training_flag = graph.get_tensor_by_name('Inputs/Training_Flag:0')
        prediction = graph.get_tensor_by_name('Network/ConvolutionnalRegressor3D/Output_Block/Fully_Connected/dense/BiasAdd:0')

        session.run(dataset.initializer)
        index = 0

        while True:
            try:
                batch_source_1, batch_source_2, batch_case = session.run(dataset.next_batch)
                batch_source_1 = np.concatenate((batch_source_1, tensor_array[np.squeeze(batch_case, axis=-1).astype(np.uint8)]), axis=-1)
                batch_prediction = session.run(prediction, feed_dict={source_1: batch_source_1, source_2: batch_source_2, training_flag: False})
                prediction_array[index] = batch_prediction[0]
                index += 1

            except tf1.errors.OutOfRangeError:
                break

    np.save(prediction_file, prediction_array)


if __name__ == '__main__':

    train()
    # predict()
