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
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def array_to_image(array, reference_image):

    data = numpy_to_vtk(np.swapaxes(array, 0, 2).ravel())
    data.SetNumberOfComponents(array.shape[-1])

    image = vtk.vtkImageData()
    image.SetDimensions(array.shape[:-1])
    image.SetOrigin(reference_image.GetOrigin())
    image.SetSpacing(reference_image.GetSpacing())
    image.GetPointData().SetScalars(data)

    return image


def image_to_array(image):

    w, h, d = image.GetDimensions()
    data = image.GetPointData().GetScalars()
    array = np.swapaxes(vtk_to_numpy(data).reshape(d, h, w, -1), 0, 2)

    return array
