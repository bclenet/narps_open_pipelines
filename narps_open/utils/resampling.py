#!/usr/bin/python
# coding: utf-8

""" A util to resample images from the narps_open project """

from os import makedirs
from os.path import dirname, join
from shutil import copy
from pathlib import Path
from argparse import ArgumentParser

from numpy import diag
from nibabel import save
from nilearn.image import load_img, resample_img

def resample_image(input_file: str, output_file: str, voxel_size_multiplier: float):
    """ Resample an image so that its voxel size is multiplied by voxel_size_multiplier

        Arguments:
            - input_file: str, path to the input file to resample
            - output_file: str, path to the output (resampled) file to create
            - voxel_size_multiplier: float, the scaling factor of a voxel size

        Example:
            An input file with a voxel size of 2mm x 2mm x 3mm, resampled with a multiplier of 2
            will end up in an image of voxel size 4mm x 4mm x 6mm.
    """
    source_image = load_img(input_file)
    voxel_sizes = source_image.header['pixdim']

    resampled_image = resample_img(
        source_image,
        interpolation = 'nearest',
        target_affine = diag((
            voxel_sizes[1] * voxel_size_multiplier,
            voxel_sizes[2] * voxel_size_multiplier,
            voxel_sizes[3] * voxel_size_multiplier))
    )

    makedirs(dirname(output_file), exist_ok = True)
    save(resampled_image, output_file)

if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser(description='Resample data from NARPS.')
    parser.add_argument('-s', '--subjects', nargs = '+', type = str, action = 'extend',
        required=True, help = 'a list of subjects')
    parser.add_argument('-m', '--multiplier', type = float,
        required=True, help = 'the voxel size multiplier')
    parser.add_argument('-d', '--dataset', type=Path,
        required=True, help='the path to the ds001734 dataset')
    parser.add_argument('-o', '--output', type=Path,
        required=True, help='the path to store the resampled dataset')
    arguments = parser.parse_args()

    dataset_dir = arguments.dataset
    resampled_dir = arguments.output

    # Parse input files (for now we only list the files we want to use later on)
    image_templates = [
        # Preprocessed func file
        'derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-MGT_run-{run}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz',
        # Confounds file
        'derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-MGT_run-{run}_bold_confounds.tsv',
        # Events file
        'sub-{subject}/func/sub-{subject}_task-MGT_run-{run}_events.tsv',
        ]

    # Perform resampling (or copy if file is not a .nii)
    run_list = ['01', '02', '03', '04']
    for subject in arguments.subjects:
        for run in run_list:
            for image in image_templates:
                if '.nii' in image:
                    # Resample nifti images
                    resample_image(
                        join(dataset_dir, image.format(subject = subject, run = run)),
                        join(resampled_dir, image.format(subject = subject, run = run)),
                        arguments.multiplier)
                else:
                    # Copy other files
                    copy(
                        join(dataset_dir, image.format(subject = subject, run = run)),
                        join(resampled_dir, image.format(subject = subject, run = run)))
