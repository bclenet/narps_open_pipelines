#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_L7J7' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_L7J7.py
    pytest -q test_team_L7J7.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import helpers, mark
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.utils.configuration import Configuration
from narps_open.pipelines.team_L7J7 import PipelineTeamL7J7

class TestPipelinesTeamL7J7:
    """ A class that contains all the unit tests for the PipelineTeamL7J7 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamL7J7 object """

        pipeline = PipelineTeamL7J7()

        # 1 - check the parameters
        assert pipeline.fwhm == 6.0
        assert pipeline.team_id == 'L7J7'

        # 2 - check workflows
        assert pipeline.get_preprocessing() is None
        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()
        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeamL7J7 object """
        pipeline = PipelineTeamL7J7()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 3, 8*2*2 + 5*2, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 0, 12, 8*2*2 + 5*2, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        info = PipelineTeamL7J7.get_subject_information([test_file, test_file])

        # Compare bunches to expected
        bunch = info[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 19.535, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [[4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param,
            [[14.0, 34.0, 38.0, 10.0, 16.0], [6.0, 14.0, 19.0, 15.0, 17.0]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

        bunch = info[1]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['gamble']
        helpers.compare_float_2d_arrays(bunch.onsets, [[4.071, 11.834, 19.535, 27.535, 36.435]])
        helpers.compare_float_2d_arrays(bunch.durations, [[4.0, 4.0, 4.0, 4.0, 4.0]])
        assert bunch.amplitudes is None
        assert bunch.tmod is None
        assert bunch.pmod[0].name == ['gain', 'loss']
        assert bunch.pmod[0].poly == [1, 1]
        helpers.compare_float_2d_arrays(bunch.pmod[0].param,
            [[14.0, 34.0, 38.0, 10.0, 16.0], [6.0, 14.0, 19.0, 15.0, 17.0]])
        assert bunch.regressor_names is None
        assert bunch.regressors is None

    @staticmethod
    @mark.unit_test
    def test_confounds_file(temporary_data_dir):
        """ Test the get_confounds_file method """

        confounds_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')
        reference_file = join(
            Configuration()['directories']['test_data'], 'pipelines', 'team_L7J7', 'confounds.tsv')

        # Get new confounds file
        PipelineTeamL7J7.get_confounds_file(confounds_file, 'sid', 'rid', temporary_data_dir)

        # Check confounds file was created
        created_confounds_file = join(
            temporary_data_dir, 'confounds_files', 'confounds_file_sub-sid_run-rid.tsv')
        assert exists(created_confounds_file)

        # Check contents
        assert cmp(reference_file, created_confounds_file)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamL7J7 and compare results """
        helpers.test_pipeline_evaluation('L7J7')
