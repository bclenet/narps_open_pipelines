#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.data.template' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_template.py
    pytest -q test_template.py -k <selected_test>
"""

from pytest import mark

from narps_open.data.template import (
    AbstractTemplateGenerator,
    RawDataTemplateGenerator,
    DerivedDataTemplateGenerator
    )

class SimpleTemplateGenerator(AbstractTemplateGenerator):
    """ A class inheriting from AbstractTemplateGenerator, for test purpose """
    def __init__(self):
        self.subject = '{subject}'
        self.verb = '{verb}'

    @property
    def variables(self) -> list:
        """ Return the list of variables handled by the generator. """
        return [self.subject, self.verb]

    @property
    def variable_names(self) -> list:
        """ Return the list of names for the variables  handled by the generator. """
        return ['subject', 'verb']

    @property
    def affirmative_sentance(self):
        """ Generate an affirmative sentance. """
        template = '{subject} is {verb}.'
        return template.format(**self.get_kwargs())

    @property
    def subject_greeting(self):
        """ Generate a subject greeting sentance. """
        template = 'Please welcome {subject}.'
        return template.format(**self.get_kwargs())

class TestAbstractTemplateGenerator:
    """ A class that contains all the unit tests for the AbstractTemplateGenerator class."""

    @staticmethod
    @mark.unit_test
    def test_use_case():
        """ Test the several usecases of a class inherited from AbstractTemplateGenerator """
        generator = SimpleTemplateGenerator()
        generator.subject = 'Paul'
        assert generator.subject_greeting == 'Please welcome Paul.'
        assert generator.affirmative_sentance == 'Paul is {verb}.'
        generator.verb = 'eating'
        assert generator.subject_greeting == 'Please welcome Paul.'
        assert generator.affirmative_sentance == 'Paul is eating.'

    @staticmethod
    @mark.unit_test
    def test_get_kwargs():
        """ Test the get_kwargs method from AbstractTemplateGenerator """

        generator = SimpleTemplateGenerator()
        assert generator.get_kwargs() == {'subject': '{subject}', 'verb': '{verb}'}
        generator.subject = 'Paul'
        assert generator.get_kwargs() == {'subject': 'Paul', 'verb': '{verb}'}
        generator.verb = 'eating'
        assert generator.get_kwargs() == {'subject': 'Paul', 'verb': 'eating'}

class TestRawDataTemplateGenerator:
    """ A class that contains all the unit tests for the RawDataTemplateGenerator class."""

    @staticmethod
    @mark.unit_test
    def test_use_case():
        """ Test the several usecases of a RawDataTemplateGenerator object """

        generator = RawDataTemplateGenerator()

        assert generator.anat_file ==\
            'sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz'
        assert generator.func_file ==\
            'sub-{subject_id}/func/sub-{subject_id}_task-{task_name}_run-{run_id}_bold.nii.gz'
        assert generator.event_file ==\
            'sub-{subject_id}/func/sub-{subject_id}_task-{task_name}_run-{run_id}_events.tsv'
        assert generator.magnitude_file ==\
            'sub-{subject_id}/fmap/sub-{subject_id}_magnitude1.nii.gz'
        assert generator.phasediff_file ==\
            'sub-{subject_id}/fmap/sub-{subject_id}_phasediff.nii.gz'

        generator.task_name = 'TT123TT'

        assert generator.anat_file ==\
            'sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz'
        assert generator.func_file ==\
            'sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run-{run_id}_bold.nii.gz'
        assert generator.event_file ==\
            'sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run-{run_id}_events.tsv'
        assert generator.magnitude_file ==\
            'sub-{subject_id}/fmap/sub-{subject_id}_magnitude1.nii.gz'
        assert generator.phasediff_file ==\
            'sub-{subject_id}/fmap/sub-{subject_id}_phasediff.nii.gz'

        generator.run_id = '01'

        assert generator.anat_file ==\
            'sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz'
        assert generator.func_file ==\
            'sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run-01_bold.nii.gz'
        assert generator.event_file ==\
            'sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run-01_events.tsv'
        assert generator.magnitude_file ==\
            'sub-{subject_id}/fmap/sub-{subject_id}_magnitude1.nii.gz'
        assert generator.phasediff_file ==\
            'sub-{subject_id}/fmap/sub-{subject_id}_phasediff.nii.gz'

        generator.subject_id = '009'

        assert generator.anat_file == 'sub-009/anat/sub-009_T1w.nii.gz'
        assert generator.func_file == 'sub-009/func/sub-009_task-TT123TT_run-01_bold.nii.gz'
        assert generator.event_file == 'sub-009/func/sub-009_task-TT123TT_run-01_events.tsv'
        assert generator.magnitude_file == 'sub-009/fmap/sub-009_magnitude1.nii.gz'
        assert generator.phasediff_file == 'sub-009/fmap/sub-009_phasediff.nii.gz'

    @staticmethod
    @mark.unit_test
    def test_defaults():
        """ Test the default setup of a DataTemplateGenerator object """

        generator = RawDataTemplateGenerator()
        generator.set_defaults()

        assert generator.anat_file == 'sub-*/anat/sub-*_T1w.nii.gz'
        assert generator.func_file == 'sub-*/func/sub-*_task-MGT_run-*_bold.nii.gz'
        assert generator.event_file == 'sub-*/func/sub-*_task-MGT_run-*_events.tsv'
        assert generator.magnitude_file == 'sub-*/fmap/sub-*_magnitude1.nii.gz'
        assert generator.phasediff_file == 'sub-*/fmap/sub-*_phasediff.nii.gz'

class TestDerivedDataTemplateGenerator:
    """ A class that contains all the unit tests for the DerivedDataTemplateGenerator class."""

    @staticmethod
    @mark.unit_test
    def test_use_case():
        """ Test the several usecases of a DerivedDataTemplateGenerator object """

        generator = DerivedDataTemplateGenerator()

        result = 'derivatives/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-{task_name}'
        result += '_run-{run_id}_bold_space-{space}_preproc.nii.gz'
        assert generator.func_preproc_file == result
        result = 'derivatives/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-{task_name}'
        result += '_run-{run_id}_bold_confounds.tsv'
        assert generator.confounds_file == result

        generator.task_name = 'TT123TT'

        result = 'derivatives/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run'
        result += '-{run_id}_bold_space-{space}_preproc.nii.gz'
        assert generator.func_preproc_file == result
        result = 'derivatives/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run'
        result += '-{run_id}_bold_confounds.tsv'
        assert generator.confounds_file == result

        generator.run_id = '01'

        result = 'derivatives/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run'
        result += '-01_bold_space-{space}_preproc.nii.gz'
        assert generator.func_preproc_file == result
        result = 'derivatives/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-TT123TT_run'
        result += '-01_bold_confounds.tsv'
        assert generator.confounds_file == result

        generator.subject_id = '009'

        result = 'derivatives/fmriprep/sub-009/func/sub-009_task-TT123TT_run'
        result += '-01_bold_space-{space}_preproc.nii.gz'
        assert generator.func_preproc_file == result
        result = 'derivatives/fmriprep/sub-009/func/sub-009_task-TT123TT_run'
        result += '-01_bold_confounds.tsv'
        assert generator.confounds_file == result

        generator.space = 'ship'

        result = 'derivatives/fmriprep/sub-009/func/sub-009_task-TT123TT_run'
        result += '-01_bold_space-ship_preproc.nii.gz'
        assert generator.func_preproc_file == result
        result = 'derivatives/fmriprep/sub-009/func/sub-009_task-TT123TT_run'
        result +='-01_bold_confounds.tsv'
        assert generator.confounds_file == result

    @staticmethod
    @mark.unit_test
    def test_defaults():
        """ Test the default setup of a DataTemplateGenerator object """

        generator = DerivedDataTemplateGenerator()
        generator.set_defaults()

        result = 'derivatives/fmriprep/sub-*/func/sub-*_task-MGT_run'
        result += '-*_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
        assert generator.func_preproc_file == result
        assert generator.confounds_file ==\
            'derivatives/fmriprep/sub-*/func/sub-*_task-MGT_run-*_bold_confounds.tsv'
