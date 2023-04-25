#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_Q6O0' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_Q6O0.py
    pytest -q test_team_Q6O0.py -k <selected_test>
"""

from statistics import mean

from pytest import raises, helpers, mark
from nipype import Workflow

from narps_open.pipelines.team_Q6O0 import PipelineTeamQ6O0

class TestPipelinesTeamQ6O0:
    """ A class that contains all the unit tests for the PipelineTeamQ6O0 class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamQ6O0 object """

        pipeline = PipelineTeamQ6O0()

        # 1 - check the parameters
        assert pipeline.fwhm == 8.0
        assert pipeline.team_id == 'Q6O0'

        # 2 - check workflows
        assert pipeline.get_preprocessing() is None
        assert pipeline.get_run_level_analysis() is None
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamQ6O0 and compare results """
        results_4_subjects = helpers.test_pipeline(
            'Q6O0',
            '/references/',
            '/data/',
            '/output/',
            4)
        assert mean(results_4_subjects) > .003
