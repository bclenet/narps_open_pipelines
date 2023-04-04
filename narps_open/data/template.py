#!/usr/bin/python
# coding: utf-8

""" Generate template strings that describe paths to data """

from os.path import join
from abc import ABC, abstractmethod

class AbstractTemplateGenerator(ABC):
    """ An abstract class to create template generators.
        This allows to handle a collection of (partially) formatted strings.
        Example:
        The string 'The {subject} is {verb}.' can be formatted with the two key arguments
        subject and verb. A template generator allow to partially format this string.
        In order to do so:
            1 - define the two arguments in the __init__() method:
                self.subject = '{subject}'
                self.verb = '{verb}'
            2 - implement the variables() method:
                def variables(self) -> list:
                    return [self.subject, self.verb]
            3 - implement the variable_names() method:
                def variable_names(self) -> list:
                    return ['subject', 'verb']
            4 - implement one property method per template string:
                @property
                def affirmative_sentance(self):
                    template = 'The {subject} is {verb}.'
                    return template.format(**self.get_kwargs())

                @property
                def negative_sentance(self):
                    template = 'The {subject} is not {verb}.'
                    return template.format(**self.get_kwargs())

                @property
                def subject_greeting(self):
                    template = 'Please welcome {subject}.'
                    return template.format(**self.get_kwargs())

            5 - these methods can later be used this way:
                generator = TemplateGenerator()
                generator.subject = 'Paul'
                print(generator.subject_greeting # Please welcome Paul.
                print(generator.affirmative_sentance) # Paul is {verb}.
                generator.verb = 'eating'
                print(generator.affirmative_sentance) # Paul is eating.
    """
    @property
    @abstractmethod
    def variables(self) -> list:
        """ Return a list of variables handled by the generator.
        The list must be ordered in the same way as the one returned
        by variable_names. """

    @property
    @abstractmethod
    def variable_names(self) -> list:
        """ Return the list of names for the variables  handled by the generator.
        The list must be ordered in the same way as the one returned
        by variable_names. """

    def get_kwargs(self) -> dict:
        """ Return a dict of keyword arguments to be passed to template
        strings to be formatted. """

        # Build a dict of variables, with their current value
        kwargs_dict = {}
        for variable, variable_name in zip(self.variables, self.variable_names):
            kwargs_dict[variable_name] = variable

        return kwargs_dict

class RawDataTemplateGenerator(AbstractTemplateGenerator):
    """ An class to create templates for original data from NARPS """

    def __init__(self):
        # Set the variables to their initial values
        self.task_name = '{task_name}'
        self.run_id = '{run_id}'
        self.subject_id = '{subject_id}'

    def set_defaults(self):
        """ Set the variables to their default values """
        self.task_name = 'MGT'
        self.run_id = '*'
        self.subject_id = '*'

    @property
    def variables(self) -> list:
        """ Return the list of variables handled by the generator. """
        return [self.run_id, self.subject_id, self.task_name]

    @property
    def variable_names(self) -> list:
        """ Return the list of names for the variables  handled by the generator. """
        return ['run_id', 'subject_id', 'task_name']

    @property
    def anat_file(self) -> str:
        """ Get the functional MRI file """
        template = join('sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')
        return template.format(**self.get_kwargs())

    @property
    def func_file(self) -> str:
        """ Get the functional MRI file """
        template = join(
            'sub-{subject_id}',
            'func',
            'sub-{subject_id}_task-{task_name}_run-{run_id}_bold.nii.gz',
        )
        return template.format(**self.get_kwargs())

    @property
    def event_file(self) -> str:
        """ Get the event file """
        template = join(
            'sub-{subject_id}',
            'func',
            'sub-{subject_id}_task-{task_name}_run-{run_id}_events.tsv',
        )
        return template.format(**self.get_kwargs())

    @property
    def magnitude_file(self) -> str:
        """ Get the magnitude file """
        template = join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_magnitude1.nii.gz')
        return template.format(**self.get_kwargs())

    @property
    def phasediff_file(self) -> str:
        """ Get the phasediff file """
        template = join('sub-{subject_id}', 'fmap', 'sub-{subject_id}_phasediff.nii.gz')
        return template.format(**self.get_kwargs())

class DerivedDataTemplateGenerator(AbstractTemplateGenerator):
    """ An class to create templates for derived data from NARPS """
    def __init__(self):
        # Set the variables to their initial values
        self.task_name = '{task_name}'
        self.space = '{space}'
        self.run_id = '{run_id}'
        self.subject_id = '{subject_id}'

    def set_defaults(self):
        """ Set the variables to their default values """
        self.task_name = 'MGT'
        self.space = 'MNI152NLin2009cAsym'
        self.run_id = '*'
        self.subject_id = '*'

    @property
    def variables(self) -> list:
        """ Return the list of variables handled by the generator. """
        return [self.run_id, self.subject_id, self.task_name, self.space]

    @property
    def variable_names(self) -> list:
        """ Return the list of names for the variables  handled by the generator. """
        return ['run_id', 'subject_id', 'task_name', 'space']

    @property
    def func_preproc_file(self) -> str:
        """ Get the preprocessed functional MRI file """
        template = join(
            'derivatives',
            'fmriprep',
            'sub-{subject_id}',
            'func',
            'sub-{subject_id}_task-{task_name}_run-{run_id}_bold_space-{space}_preproc.nii.gz',
        )
        return template.format(**self.get_kwargs())

    @property
    def confounds_file(self) -> str:
        """ Get the confounds file """
        template = join(
            'derivatives',
            'fmriprep',
            'sub-{subject_id}',
            'func',
            'sub-{subject_id}_task-{task_name}_run-{run_id}_bold_confounds.tsv',
        )
        return template.format(**self.get_kwargs())
