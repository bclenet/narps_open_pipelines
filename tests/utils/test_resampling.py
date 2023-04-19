#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils.resampling' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_resampling.py
    pytest -q test_resampling.py -k <selected_test>
"""

from os import makedirs
from os.path import join
from filecmp import cmp

from pytest import mark

from narps_open.utils.configuration import Configuration
from narps_open.utils.resampling import resample_image

class TestResampling:
    """ A class that contains all the unit tests for the resampling module."""

    @staticmethod
    @mark.unit_test
    def test_resample():
        """ Test the resample_image function """
        input_file = join(
            Configuration()['directories']['test_data'],
            'utils','resampling',
            'input_file.nii.gz'
            )
        output_file_4 = join(
            Configuration()['directories']['test_runs'],
            'utils','resampling',
            'test_resample_4.nii.gz'
            )
        reference_file_4 = join(
            Configuration()['directories']['test_data'],
            'utils','resampling',
            'reference_file_4.nii.gz'
            )
        output_file_05 = join(
            Configuration()['directories']['test_runs'],
            'utils','resampling',
            'test_resample_05.nii.gz'
            )
        reference_file_05 = join(
            Configuration()['directories']['test_data'],
            'utils','resampling',
            'reference_file_05.nii.gz'
            )

        makedirs(dirname(output_file_4), exist_ok = True)
        resample_image(input_file, output_file_4, 4)
        assert cmp(output_file_4, reference_file_4)

        makedirs(dirname(output_file_05), exist_ok = True)
        resample_image(input_file, output_file_05, .5)
        assert cmp(output_file_05, reference_file_05)
