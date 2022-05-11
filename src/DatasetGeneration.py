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


import json
import matplotlib.pyplot as plt
import numpy as np
import random
import SimpleITK as sitk
import vtk
from os.path import join
from tgstk import tgstk
from vtk_numpy_utils import array_to_image, image_to_array


def generate_tumors():

    # Generates a random synthetic tumor dataset from the pre-processed MRI data of healthy subjects

    # Base directory containing the pre-processed MRI data, which can be found at
    # https://lisaserver.ulb.ac.be/owncloud/index.php/s/KwEPG65gh1U7xNM.
    base_dir = ''

    # Output directory for the synthetic tumors
    output_dir = ''

    # Subdirectories for each subject
    dirs = ['01',
            '02',
            '03',
            '04',
            '05',
            '06']

    # Corresponding volumes of interest with dimensions 192x192x128 centered on the brain
    vois = [(slice(29, 221), slice(34, 226), slice(16, 144)),
            (slice(26, 218), slice(22, 214), slice(15, 143)),
            (slice(26, 218), slice(33, 225), slice(14, 142)),
            (slice(25, 217), slice(36, 228), slice(17, 145)),
            (slice(25, 217), slice(21, 213), slice(16, 144)),
            (slice(22, 214), slice(21, 213), slice(14, 142))]

    # Slice ranges for tumor seeds
    seed_ranges = [(60, 120),
                   (62, 122),
                   (60, 120),
                   (63, 123),
                   (62, 122),
                   (56, 116)]

    dt = 0.5  # Time step
    n = 200   # Number of synthetic tumors per subject

    parameters = []  # Sampled parameters

    for i in range(len(dirs)):

        dir = join(base_dir, dirs[i])
        voi = vois[i]
        seed_range = seed_ranges[i]

        # Brain map
        brain_map_file = join(dir, 'brain_domain.mha')
        brain_map_reader = vtk.vtkMetaImageReader()
        brain_map_reader.SetFileName(brain_map_file)
        brain_map_reader.Update()
        brain_map_image = brain_map_reader.GetOutput()
        brain_map_array = image_to_array(brain_map_image)

        # Seeds
        white_matter_array = (brain_map_array == tgstk.WHITE_MATTER)
        white_matter_array[:, :, :seed_range[0], 0] = False
        white_matter_array[:, :, seed_range[1]:, 0] = False

        white_matter_image = sitk.GetImageFromArray(white_matter_array[:, :, :, 0].astype(np.uint8))
        erosion_filter = sitk.BinaryErodeImageFilter()  # Guarantees a 3x3x3 white matter neighborhood
        erosion_filter.SetForegroundValue(1)
        erosion_filter.SetKernelType(sitk.sitkBox)
        erosion_filter.SetKernelRadius(1)
        white_matter_array = sitk.GetArrayFromImage(erosion_filter.Execute(white_matter_image))[..., np.newaxis]
        white_matter_voxels = list(zip(*np.where(white_matter_array)))

        # Diffusion tensor
        unit_diffusion_tensor_file = join(dir, 'unit_diffusion_tensor.mha')
        unit_tensor_reader = vtk.vtkMetaImageReader()
        unit_tensor_reader.SetFileName(unit_diffusion_tensor_file)
        unit_tensor_reader.Update()
        unit_tensor_image = unit_tensor_reader.GetOutput()
        unit_tensor_array = (1.0/365.0)*image_to_array(unit_tensor_image)  # Includes the yr to d conversion factor

        # Proliferation rate
        unit_proliferation_rate_file = join(dir, 'unit_proliferation_rate.mha')
        unit_proliferation_rate_reader = vtk.vtkMetaImageReader()
        unit_proliferation_rate_reader.SetFileName(unit_proliferation_rate_file)
        unit_proliferation_rate_reader.Update()
        unit_proliferation_rate_image = unit_proliferation_rate_reader.GetOutput()
        unit_proliferation_rate_array = (1.0/365.0)*image_to_array(unit_proliferation_rate_image)  # Includes the yr to d conversion factor

        # Parameter distribution
        l_min = np.sqrt(10.0/40.0)  # [mm]
        l_max = np.sqrt(40.0/10.0)  # [mm]
        v_min = 2.0*np.sqrt(200.0)  # [mm/yr]
        v_max = 2.0*np.sqrt(600.0)  # [mm/yr]

        # Simulated times
        t1_min = 90   # [d]
        t1_max = 180  # [d]
        t2_min = 90   # [d]
        t2_max = 180  # [d]
        t3 = 90       # [d]
        t4 = 90       # [d]

        # Time step check for numerical stability
        spacing = [1.0, 1.0, 1.0]  # [mm]
        d_max = l_max*v_max/2.0    # [mm²/yr]
        dt_max = np.min(0.5/(d_max*(unit_tensor_array[:, :, :, 0]/spacing[0]**2 +
                                    unit_tensor_array[:, :, :, 3]/spacing[1]**2 +
                                    unit_tensor_array[:, :, :, 5]/spacing[2]**2)))  # [d]
        if dt > dt_max:
            print(f'Warning: The specified time step ({dt} days) is larger than the maximum time step allowed ({dt_max} days).')

        # Simulation
        n_1 = 0  # Number of generated tumors
        n_2 = 0  # Number of valid tumors (with non empty imaging contours)

        random.seed(1993)

        while n_2 != n:

            n_1 += 1

            # Random parameter picking
            l = random.uniform(l_min, l_max)
            v = random.uniform(v_min, v_max)
            white_matter_diffusion_rate = l*v/2.0
            proliferation_rate = v/(2.0*l)

            seed = random.choice(white_matter_voxels)
            t1 = random.randint(t1_min, t1_max)  # First imaging time
            t2 = random.randint(t2_min, t2_max)  # Second imaging time

            # Initial density
            initial_cell_density_array = np.zeros_like(brain_map_array, dtype=np.float64)
            initial_cell_density_array[seed[0]-1:seed[0]+2, seed[1]-1:seed[1]+2, seed[2]-1:seed[2]+2, 0] = 1.0  # 3x3x3 neighborhood centered on the tumor seed
            initial_cell_density_image = array_to_image(initial_cell_density_array, brain_map_image)

            # Proliferation rate
            proliferation_rate_array = proliferation_rate*unit_proliferation_rate_array
            proliferation_rate_image = array_to_image(proliferation_rate_array, brain_map_image)

            # Diffusion tensor
            diffusion_tensor_array = white_matter_diffusion_rate*unit_tensor_array
            diffusion_tensor_image = array_to_image(diffusion_tensor_array, brain_map_image)

            # Parameters
            p = {'Seed': (int(seed[0]-voi[0].start), int(seed[1]-voi[1].start), int(seed[2]-voi[2].start)),  # VOI offset subtraction for seed coordinates
                 'Proliferation rate': float(proliferation_rate),
                 'Diffusivity': float(white_matter_diffusion_rate),
                 'Infiltration length': float(l),
                 'Propagation velocity': float(v),
                 'T1': int(t1),
                 'T2': int(t2),
                 'T3': int(t3),
                 'T4': int(t4)}

            # Tumor growth simulation from emergence to the first imaging time
            filter = tgstk.tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter()
            filter.setBrainMapImage(brain_map_image)
            filter.setDiffusionTensorImage(diffusion_tensor_image)
            filter.setInitialCellDensityImage(initial_cell_density_image)
            filter.setProliferationRateImage(proliferation_rate_image)
            filter.setSimulatedTime(t1)
            filter.setTimeStep(dt)
            filter.update()
            final_cell_density_image_1 = filter.getFinalCellDensityImage()
            final_cell_density_array_1 = image_to_array(final_cell_density_image_1)
            roi_1 = final_cell_density_array_1 > 0.16  # Gamma_3 contour

            # Tumor growth simulation from the first to the second imaging time
            filter.setInitialCellDensityImage(final_cell_density_image_1)
            filter.setSimulatedTime(t2)
            filter.update()
            final_cell_density_image_2 = filter.getFinalCellDensityImage()
            final_cell_density_array_2 = image_to_array(final_cell_density_image_2)
            roi_2 = final_cell_density_array_2 > 0.80  # Gamma_1 contour
            roi_3 = final_cell_density_array_2 > 0.16  # Gamma_2 contour

            # Tumor growth simulation from the second to the third imaging time
            filter.setInitialCellDensityImage(final_cell_density_image_2)
            filter.setSimulatedTime(t3)
            filter.update()
            final_cell_density_image_3 = filter.getFinalCellDensityImage()
            final_cell_density_array_3 = image_to_array(final_cell_density_image_3)

            # Tumor growth simulation from the third to the fourth imaging time
            filter.setInitialCellDensityImage(final_cell_density_image_3)
            filter.setSimulatedTime(t4)
            filter.update()
            final_cell_density_image_4 = filter.getFinalCellDensityImage()
            final_cell_density_array_4 = image_to_array(final_cell_density_image_4)

            # Plot synthetic tumor
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_1[:, :, seed[2], 0]), vmin=0.0, vmax=1.0, cmap='jet')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_2[:, :, seed[2], 0]), vmin=0.0, vmax=1.0, cmap='jet')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_3[:, :, seed[2], 0]), vmin=0.0, vmax=1.0, cmap='jet')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_1[:, :, seed[2], 0] > 0.16))
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_2[:, :, seed[2], 0] > 0.80))
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_2[:, :, seed[2], 0] > 0.16))
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_1[:, seed[1], :, 0]), vmin=0.0, vmax=1.0, cmap='jet', origin='lower')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_2[:, seed[1], :, 0]), vmin=0.0, vmax=1.0, cmap='jet', origin='lower')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_3[:, seed[1], :, 0]), vmin=0.0, vmax=1.0, cmap='jet', origin='lower')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_1[:, seed[1], :, 0] > 0.16), origin='lower')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_2[:, seed[1], :, 0] > 0.80), origin='lower')
            # plt.figure()
            # plt.imshow(np.transpose(final_cell_density_array_2[:, seed[1], :, 0] > 0.16), origin='lower')
            # plt.show()

            # Check for empty contours
            if np.count_nonzero(roi_1) != 0 and np.count_nonzero(roi_2) != 0 and np.count_nonzero(roi_3) != 0:

                n_2 += 1  # Valid tumor
                index = int(np.log10(n) - np.floor(np.log10(n_2))) * '0' + str(n_2)

                # VOI extraction
                t1_array = final_cell_density_array_1[voi].astype(np.float32)
                t2_array = final_cell_density_array_2[voi].astype(np.float32)
                t3_array = final_cell_density_array_3[voi].astype(np.float32)
                t4_array = final_cell_density_array_4[voi].astype(np.float32)

                # Saving arrays
                np.save(join(output_dir, join('t1', f'0{i+1}_{index}.npy')), t1_array)
                np.save(join(output_dir, join('t2', f'0{i+1}_{index}.npy')), t2_array)
                np.save(join(output_dir, join('t3', f'0{i+1}_{index}.npy')), t3_array)
                np.save(join(output_dir, join('t4', f'0{i+1}_{index}.npy')), t4_array)

                # Saving parameters
                parameters.append(p)
                with open(join(output_dir, 'Parameters.json'), 'w+') as f:
                     json.dump(parameters, f)

            else:
                print('Empty ROI!')

            # Force garbage collection for SWIG shared objects
            del initial_cell_density_image
            del initial_cell_density_array
            del proliferation_rate_image
            del proliferation_rate_array
            del diffusion_tensor_image
            del diffusion_tensor_array
            del final_cell_density_image_1
            del final_cell_density_array_1
            del final_cell_density_image_2
            del final_cell_density_array_2
            del final_cell_density_image_3
            del final_cell_density_array_3
            del filter

        # Force garbage collection for SWIG shared objects
        del brain_map_array
        del brain_map_image
        del unit_tensor_array
        del unit_tensor_image


def generate_distribution_dataset():

    # Generates the tumor cell density distribution dataset from the synthetic tumor generated by generate_tumors()

    # Base directory containing the pre-processed MRI data, which can be found at
    # https://lisaserver.ulb.ac.be/owncloud/index.php/s/KwEPG65gh1U7xNM.
    base_dir = ''

    # Directory containing the tumors generated by generate_tumors()
    tumor_dir = ''

    shape = (192, 192, 128)  # VOI shape
    cases = 6                # Number of subjects
    tumours_per_case = 200   # Number of synthetic tumors per subject
    testing_cases = [5]      # Test subjects

    # Volumes of interest with dimensions 192x192x128 centered on the brain for each subject
    vois = [(slice(29, 221), slice(34, 226), slice(16, 144)),
            (slice(26, 218), slice(22, 214), slice(15, 143)),
            (slice(26, 218), slice(33, 225), slice(14, 142)),
            (slice(25, 217), slice(36, 228), slice(17, 145)),
            (slice(25, 217), slice(21, 213), slice(16, 144)),
            (slice(22, 214), slice(21, 213), slice(14, 142))]

    training_source_array = np.zeros(((cases-len(testing_cases))*tumours_per_case,) + shape + (2,), dtype=np.float32)  # Training source imaging contours
    training_target_array = np.zeros(((cases-len(testing_cases))*tumours_per_case,) + shape + (1,), dtype=np.float32)  # Training target density distributions
    training_case_array = np.zeros(((cases-len(testing_cases))*tumours_per_case, 1), dtype=np.uint8)                   # Subject IDs for each training tumor to retrieve the corresponding diffusion tensor
    training_tensor_array = np.zeros((cases-len(testing_cases),) + shape + (6,), dtype=np.float32)                     # Diffusion tensors of the training subjects (stored only once per subject for memory efficiency)
    testing_source_array = np.zeros((len(testing_cases)*tumours_per_case,) + shape + (2,), dtype=np.float32)           # Testing source imaging contours
    testing_target_array = np.zeros((len(testing_cases)*tumours_per_case,) + shape + (1,), dtype=np.float32)           # Testing target density distributions
    testing_case_array = np.zeros((len(testing_cases)*tumours_per_case, 1), dtype=np.uint8)                            # Subject IDs for each testing tumor to retrieve the corresponding diffusion tensor
    testing_tensor_array = np.zeros((len(testing_cases),) + shape + (6,), dtype=np.float32)                            # Diffusion tensors of the testing subjects (stored only once per subject for memory efficiency)

    threshold_1 = 0.80  # c_1
    threshold_2 = 0.16  # c_2

    training_tumor_index = 0
    testing_tumor_index = 0
    training_case_index = 0
    testing_case_index = 0

    for c in range(cases):

        a = max(int(np.log10(cases)-np.floor(np.log10(c+1))), 1)*'0' + str(c+1)  # Subject

        for t in range(tumours_per_case):

            b = int(np.log10(tumours_per_case)-np.floor(np.log10(t+1)))*'0' + str(t+1)  # Tumor
            file = f'{a}_{b}.npy'
            cell_density_array = np.load(join(tumor_dir, join('t2', file)))  # Second imaging time

            if c in testing_cases:
                testing_source_array[testing_tumor_index, :, :, :, 0] = (cell_density_array > threshold_1)[:, :, :, 0].astype(np.float32)  # Gamma_1 contour
                testing_source_array[testing_tumor_index, :, :, :, 1] = (cell_density_array > threshold_2)[:, :, :, 0].astype(np.float32)  # Gamma_2 contour
                testing_target_array[testing_tumor_index] = cell_density_array.astype(np.float32)
                testing_case_array[testing_tumor_index] = testing_case_index
                testing_tumor_index += 1

            else:
                training_source_array[training_tumor_index, :, :, :, 0] = (cell_density_array > threshold_1)[:, :, :, 0].astype(np.float32)  # Gamma_1 contour
                training_source_array[training_tumor_index, :, :, :, 1] = (cell_density_array > threshold_2)[:, :, :, 0].astype(np.float32)  # Gamma_2 contour
                training_target_array[training_tumor_index] = cell_density_array.astype(np.float32)
                training_case_array[training_tumor_index] = training_case_index
                training_tumor_index += 1

        tensor_image = sitk.ReadImage(join(base_dir, join(a, 'unit_diffusion_tensor.mha')))
        tensor_array = (1.0/365.0)*np.swapaxes(sitk.GetArrayViewFromImage(tensor_image), 0, 2)  # sitk returns (z, y, x) arrays => axes are swapped
        voi = vois[c]

        if c in testing_cases:
            testing_tensor_array[testing_case_index] = tensor_array[voi].astype(np.float32)
            testing_case_index += 1

        else:
            training_tensor_array[training_case_index] = tensor_array[voi].astype(np.float32)
            training_case_index += 1

    # Saving arrays
    np.save(join(tumor_dir, join('distribution', 'Training/Source.npy')), training_source_array)
    np.save(join(tumor_dir, join('distribution', 'Training/Target.npy')), training_target_array)
    np.save(join(tumor_dir, join('distribution', 'Training/Case.npy')), training_case_array)
    np.save(join(tumor_dir, join('distribution', 'Training/Tensor.npy')), training_tensor_array)
    np.save(join(tumor_dir, join('distribution', 'Testing/Source.npy')), testing_source_array)
    np.save(join(tumor_dir, join('distribution', 'Testing/Target.npy')), testing_target_array)
    np.save(join(tumor_dir, join('distribution', 'Testing/Case.npy')), testing_case_array)
    np.save(join(tumor_dir, join('distribution', 'Testing/Tensor.npy')), testing_tensor_array)


def generate_parameter_dataset():

    # Generates the model parameter dataset from the synthetic tumor generated by generate_tumors()

    # Directory containing the tumors generated by generate_tumors()
    tumor_dir = ''

    shape = (192, 192, 128)  # VOI shape
    cases = 6                # Number of subjects
    tumours_per_case = 200   # Number of synthetic tumors per subject
    testing_cases = [5]      # Test subjects

    # training_case_array, training_tensor_array, testing_case_array, and testing_tensor_array must be retrieved from
    # the density distribution dataset.
    training_source_array_1 = np.zeros(((cases-len(testing_cases))*tumours_per_case,) + shape + (3,), dtype=np.float32)  # Training source imaging contours
    training_source_array_2 = np.zeros(((cases-len(testing_cases))*tumours_per_case,) + (1,), dtype=np.float32)          # Training source time intervals
    training_target_array = np.zeros(((cases-len(testing_cases))*tumours_per_case,) + (2,), dtype=np.float32)            # Training target parameter values
    testing_source_array_1 = np.zeros((len(testing_cases)*tumours_per_case,) + shape + (3,), dtype=np.float32)           # Testing source imaging contours
    testing_source_array_2 = np.zeros((len(testing_cases)*tumours_per_case,) + (1,), dtype=np.float32)                   # Testing source time intervals
    testing_target_array = np.zeros((len(testing_cases)*tumours_per_case,) + (2,), dtype=np.float32)                     # Testing target parameter values

    threshold_1 = 0.80  # c_1
    threshold_2 = 0.16  # c_2

    index = 0
    training_index = 0
    testing_index = 0

    with open(join(tumor_dir, 'Parameters.json'), 'r') as f:
        parameters = json.load(f)

    for c in range(cases):

        a = max(int(np.log10(cases)-np.floor(np.log10(c+1))), 1)*'0' + str(c+1)  # Subject

        for t in range(tumours_per_case):

            b = int(np.log10(tumours_per_case)-np.floor(np.log10(t+1)))*'0' + str(t+1)  # Tumor
            file = f'{a}_{b}.npy'
            cell_density_array_1 = np.load(join(tumor_dir, join('t1', file)))  # First imaging time
            cell_density_array_2 = np.load(join(tumor_dir, join('t2', file)))  # Second imaging time
            l = parameters[index]['Infiltration length']
            v = parameters[index]['Propagation velocity']

            if c in testing_cases:
                testing_source_array_1[testing_index, :, :, :, 0] = (cell_density_array_2 > threshold_1)[:, :, :, 0].astype(np.float32)  # Gamma_1 contour
                testing_source_array_1[testing_index, :, :, :, 1] = (cell_density_array_2 > threshold_2)[:, :, :, 0].astype(np.float32)  # Gamma_2 contour
                testing_source_array_1[testing_index, :, :, :, 2] = (cell_density_array_1 > threshold_2)[:, :, :, 0].astype(np.float32)  # Gamma_3 contour
                testing_source_array_2[testing_index] = parameters[index]['T2']
                testing_target_array[testing_index] = np.array([l, v], dtype=np.float32)
                testing_index += 1

            else:
                training_source_array_1[training_index, :, :, :, 0] = (cell_density_array_2 > threshold_1)[:, :, :, 0].astype(np.float32)  # Gamma_1 contour
                training_source_array_1[training_index, :, :, :, 1] = (cell_density_array_2 > threshold_2)[:, :, :, 0].astype(np.float32)  # Gamma_2 contour
                training_source_array_1[training_index, :, :, :, 2] = (cell_density_array_1 > threshold_2)[:, :, :, 0].astype(np.float32)  # Gamma_3 contour
                training_source_array_2[training_index] = parameters[index]['T2']
                training_target_array[training_index] = np.array([l, v], dtype=np.float32)
                training_index += 1

            index += 1

    np.save(join(tumor_dir, join('parameters', 'Training/Source_1.npy')), training_source_array_1)
    np.save(join(tumor_dir, join('parameters', 'Training/Source_2.npy')), training_source_array_2)
    np.save(join(tumor_dir, join('parameters', 'Training/Target.npy')), training_target_array)
    np.save(join(tumor_dir, join('parameters', 'Testing/Source_1.npy')), testing_source_array_1)
    np.save(join(tumor_dir, join('parameters', 'Testing/Source_2.npy')), testing_source_array_2)
    np.save(join(tumor_dir, join('parameters', 'Testing/Target.npy')), testing_target_array)


if __name__ == '__main__':

    generate_tumors()
    generate_distribution_dataset()
    generate_parameter_dataset()
