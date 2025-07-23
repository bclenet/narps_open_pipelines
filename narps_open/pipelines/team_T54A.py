#!/usr/bin/python
# coding: utf-8

""" Write the work of NARPS team T54A using Nipype """

from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.utility import IdentityInterface, Function, Split, Merge
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl.aroma import ICA_AROMA
from nipype.interfaces.fsl.preprocess import SUSAN, BET
from nipype.interfaces.fsl.model import (
    Level1Design, FEATModel, FLAMEO, FILMGLS, MultipleRegressDesign,
    FSLCommand, SmoothEstimate, Randomise
    )
from nipype.interfaces.fsl.utils import ImageMaths, ImageStats, Merge as FSLMerge

from narps_open.utils.configuration import Configuration
from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import list_intersection, elements_in_string, clean_list
from narps_open.core.interfaces import InterfaceFactory

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeamT54A(Pipeline):
    """ A class that defines the pipeline of team T54A """

    def __init__(self):
        super().__init__()
        self.fwhm = 4.0
        self.team_id = 'T54A'
        self.contrast_list = ['1', '2']
        self.run_level_contrasts = [
            ('gain', 'T', ['trial', 'gain', 'loss'], [0, 1, 0]),
            ('loss', 'T', ['trial', 'gain', 'loss'], [0, 0, 1])
            ]

    def get_motion_parameters(in_file):
        """
        Create a tsv file with only motion parameters per subject per run.

        Parameters :
        - in_file : path to the subject parameters file (i.e. one per run)

        Return :
        - str : path to new file containing only motion parameters.
        """
        from os.path import abspath

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        # Extract motion parameters
        data_frame = read_csv(in_file, sep = '\t', header=0)
        retained_parameters = DataFrame(transpose(array([
            data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']
            ])))

        # Write dataframe
        parameters_file = abspath('motion_parameters.tsv')
        with open(parameters_file, 'w') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return parameters_file

    def get_preprocessing(self):
        """ Create the preprocessing workflow """

        # Initiate the workflow
        preprocessing = Workflow(
            base_dir = self.directories.working_dir,
            name = 'preprocessing'
            )

        # IdentityInterface Node - Iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - Get necessary files
        templates = {
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        preprocessing.connect(information_source, 'subject_id', select_files, 'subject_id')
        preprocessing.connect(information_source, 'run_id', select_files, 'run_id')

        # ImageMaths - Convert func to float representation
        func_to_float = Node(ImageMaths(), name = 'func_to_float')
        func_to_float.inputs.out_data_type = 'float'
        func_to_float.inputs.op_string = ''
        func_to_float.inputs.suffix = '_dtype'
        preprocessing.connect(select_files, 'func', func_to_float, 'in_file')

        # Function get_motion_parameters - Extract motion parameters from confounds file
        motion_parameters = Node(Function(
            function = self.get_motion_parameters,
            input_names = ['in_file'],
            output_names = ['out_file'],
            ), name = 'motion_parameters')
        preprocessing.connect(select_files, 'confounds', motion_parameters, 'in_file')

        # ICA_AROMA - Noise estimation of func image
        noise_estimation = Node(ICA_AROMA(), name = 'noise_estimation')
        preprocessing.connect(func_to_float, 'out_file', noise_estimation, 'in_file')
        preprocessing.connect(motion_parameters, 'out_file', noise_estimation, 'motion_parameters')

        # BET - Extract brain from func
        skull_stripping_func = Node(BET(), name = 'skull_stripping_func')
        skull_stripping_func.inputs.frac = 0.3
        skull_stripping_func.inputs.functional = True
        skull_stripping_func.inputs.mask = True
        skull_stripping_func.inputs.threshold = True
        preprocessing.connect(
            noise_estimation, 'nonaggr_denoised_file', skull_stripping_func, 'in_file')

        # ImageMaths - Compute the mean image of each time point
        mean_func = Node(ImageMaths(), name = 'mean_func')
        mean_func.inputs.suffix = '_mean'
        mean_func.inputs.op_string = '-Tmean'
        preprocessing.connect(skull_stripping_func, 'out_file', mean_func, 'in_file')

        # ImageStats - Compute the median value of each time point
        # (only because it's needed by SUSAN)
        median_value = Node(ImageStats(), name = 'median_value')
        median_value.inputs.op_string = '-k %s -p 50'
        preprocessing.connect(skull_stripping_func, 'out_file', median_value, 'in_file')
        preprocessing.connect(skull_stripping_func, 'mask_file', median_value, 'mask_file')

        # Merge - Merge the median values with the mean functional images into a coupled list
        merge_median = Node(Merge(2), name = 'merge_median')
        preprocessing.connect(mean_func, 'out_file', merge_median, 'in1')
        preprocessing.connect(median_value, 'out_stat', merge_median, 'in2')

        # SUSAN - Smoothing funk
        smooth_func = Node(SUSAN(), name = 'smooth_func')
        smooth_func.inputs.fwhm = self.fwhm

        # Define a function to get the brightness threshold for SUSAN
        get_brightness_threshold = lambda median : 0.75 * median

        # Define a function to get the usans for SUSAN
        get_usans = lambda value : [tuple([value[0], 0.75 * value[1]])]

        preprocessing.connect(skull_stripping_func, 'out_file', smooth_func, 'in_file')
        preprocessing.connect(
            median_value, ('out_stat', get_brightness_threshold),
            smooth_func, 'brightness_threshold')
        preprocessing.connect(merge_median, ('out', get_usans), smooth_func, 'usans')

        # Define a function to get the scaling factor for intensity normalization
        get_intensity_normalization_scale = lambda median : '-mul %.10f' % (10000. / median)

        # ImageMaths - Scale each time point so that its median value is 10000
        normalize_intensity = Node(ImageMaths(), name = 'normalize_intensity')
        normalize_intensity.inputs.suffix = '_intnorm'
        preprocessing.connect(smooth_func, 'smoothed_file', normalize_intensity, 'in_file')
        preprocessing.connect(
            median_value, ('out_stat', get_intensity_normalization_scale),
            normalize_intensity, 'op_string')

        # ImageMaths - Generate a mean functional image from the scaled data
        mean_func_2 = Node(ImageMaths(), name = 'mean_func_2')
        mean_func_2.inputs.op_string = '-Tmean'
        preprocessing.connect(normalize_intensity, 'out_file', mean_func_2, 'in_file')

        # ImageMaths - Perform temporal highpass filtering on the data
        def get_high_pass_filter_command(in_file):
            """ Create command line for high pass filtering using image maths """
            from narps_open import TaskInformation

            high_pass_filter_cutoff = 100 #seconds
            repetition_time = float(TaskInformation()['RepetitionTime'])

            return f'-bptf {high_pass_filter_cutoff / (2.0 * repetition_time)} -1 -add {in_file}'

        high_pass_filter = Node(ImageMaths(), name = 'high_pass_filter')
        high_pass_filter.inputs.suffix = '_tempfilt'
        preprocessing.connect(normalize_intensity, 'out_file', high_pass_filter, 'in_file')
        preprocessing.connect(
            mean_func_2, ('out_file', get_high_pass_filter_command),
            high_pass_filter, 'op_string')

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        preprocessing.connect(
            high_pass_filter, 'out_file', data_sink, 'preprocessing.@filtered_file')

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:

            # Merge Node - Merge func file names to be removed after datasink node is performed
            merge_removable_files = Node(Merge(7), name = 'merge_removable_files')
            merge_removable_files.inputs.ravel_inputs = True
            preprocessing.connect(func_to_float, 'out_file', merge_removable_files, 'in1')
            preprocessing.connect(noise_estimation, 'nonaggr_denoised_file', merge_removable_files, 'in2')
            preprocessing.connect(skull_stripping_func, 'out_file', merge_removable_files, 'in3')
            preprocessing.connect(smooth_func, 'smoothed_file', merge_removable_files, 'in4')
            preprocessing.connect(normalize_intensity, 'out_file', merge_removable_files, 'in5')
            preprocessing.connect(mean_func, 'out_file', merge_removable_files, 'in6')
            preprocessing.connect(mean_func_2, 'out_file', merge_removable_files, 'in7')

            # Function Nodes remove_files - Remove sizeable func files once they aren't needed
            remove_dirs = MapNode(Function(
                function = remove_parent_directory,
                input_names = ['_', 'file_name'],
                output_names = []
                ), name = 'remove_dirs', iterfield = 'file_name')
            preprocessing.connect(merge_removable_files, 'out', remove_dirs, 'file_name')
            preprocessing.connect(data_sink, 'out_file', remove_dirs, '_')

        return preprocessing

    def get_preprocessing_outputs(self):
        """ Return the names of the files the preprocessing is supposed to generate. """

        # Outputs from preprocessing (intensity normalized files)
        parameters = {
            'subject_id': self.subject_list,
            'run_id': self.run_list,
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir, 'preprocessing',
            '_run_id_{run_id}_subject_id_{subject_id}',
            'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_dtype_thresh_smooth_intnorm.nii.gz')

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

    def get_subject_information(event_file):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : str, file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from nipype.interfaces.base import Bunch

        condition_names = ['trial', 'gain', 'loss', 'difficulty', 'response', 'missed']
        onsets = {}
        durations = {}
        amplitudes = {}

        for condition in condition_names:
            # Create dictionary items with empty lists
            onsets.update({condition : []})
            durations.update({condition : []})
            amplitudes.update({condition : []})

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()

                if info[5] != 'NoResp':
                    onsets['trial'].append(float(info[0]))
                    durations['trial'].append(float(info[4]))
                    amplitudes['trial'].append(1.0)
                    onsets['gain'].append(float(info[0]))
                    durations['gain'].append(float(info[4]))
                    amplitudes['gain'].append(float(info[2]))
                    onsets['loss'].append(float(info[0]))
                    durations['loss'].append(float(info[4]))
                    amplitudes['loss'].append(float(info[3]))
                    onsets['difficulty'].append(float(info[0]))
                    durations['difficulty'].append(float(info[4]))
                    amplitudes['difficulty'].append(
                        abs(0.5 * float(info[2]) - float(info[3]))
                        )
                    onsets['response'].append(float(info[0]) + float(info[4]))
                    durations['response'].append(0.0)
                    amplitudes['response'].append(1.0)
                else:
                    onsets['missed'].append(float(info[0]))
                    durations['missed'].append(0.0)
                    amplitudes['missed'].append(1.0)

        # Check if there where missed trials for this run
        if not onsets['missed']:
            condition_names.remove('missed')

        return [
            Bunch(
                conditions = condition_names,
                onsets = [onsets[k] for k in condition_names],
                durations = [durations[k] for k in condition_names],
                amplitudes = [amplitudes[k] for k in condition_names],
                regressor_names = None,
                regressors = None)
            ]

    def get_parameters_file(in_file):
        """
        Create a tsv file with only desired parameters per subject per run.

        Parameters :
        - filepath : path to the subject parameters file (i.e. one per run)
        - subject_id : subject for whom the 1st level analysis is made
        - run_id: run for which the 1st level analysis is made
        - working_dir: str, name of the directory for intermediate results

        Return :
        - parameters_file : paths to new files containing only desired parameters.
        """
        from os.path import abspath

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        data_frame = read_csv(filepath, sep = '\t', header=0)
        if 'NonSteadyStateOutlier00' in data_frame.columns:
            temp_list = array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ'],
                data_frame['NonSteadyStateOutlier00']])
        else:
            temp_list = array([
                data_frame['X'], data_frame['Y'], data_frame['Z'],
                data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']])
        retained_parameters = DataFrame(transpose(temp_list))

        parameters_file = abspath(f'parameters_file.tsv')

        with open(parameters_file, 'w') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return parameters_file

    def get_run_level_analysis(self):
        """
        Create the run level analysis workflow.

        Returns:
            - run_level_analysis : nipype.WorkFlow
        """
        # Create run level analysis workflow and connect its nodes
        run_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )
        
        # IdentityInterface Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - to select necessary files
        template = {
            # Parameter file
            'confounds' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_confounds.tsv'),
            # Preprocessed functional MRI
            'func' : join(self.directories.output_dir, 'preprocessing',
                '_run_id_{run_id}_subject_id_{subject_id}',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc_dtype_thresh_smooth_intnorm.nii.gz'
            ),
            # Event file
            'events' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv')
        }
        select_files = Node(SelectFiles(template), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        run_level_analysis.connect(information_source, 'subject_id', select_files, 'subject_id')
        run_level_analysis.connect(information_source, 'run_id', select_files, 'run_id')

        # Function Node get_subject_infos - Get subject specific condition information
        subject_information = Node(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info']
            ), name = 'subject_information')
        run_level_analysis.connect(select_files, 'events', subject_information, 'event_file')

        # Function Node get_parameters_file - Get files with movement parameters
        parameters = Node(Function(
            function = self.get_parameters_file,
            input_names = ['in_file'],
            output_names = ['parameters_file']),
            name = 'parameters')
        parameters.inputs.working_dir = self.directories.working_dir
        run_level_analysis.connect(select_files, 'confounds', parameters, 'in_file')

        # SpecifyModel Node - Generate run level model
        specify_model = Node(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 100
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        run_level_analysis.connect(
            parameters, 'parameters_file', specify_model, 'realignment_parameters')
        run_level_analysis.connect(
            subject_information, 'subject_info', specify_model, 'subject_info')
        run_level_analysis.connect(select_files, 'func', specify_model, 'functional_runs')

        # Level1Design Node - Generate files for run level computation
        model_design = Node(Level1Design(), name = 'model_design')
        model_design.inputs.bases = {'dgamma':{'derivs' : True}}
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contrasts
        run_level_analysis.connect(specify_model, 'session_info', model_design, 'session_info')

        # FEATModel Node - Generate run level model
        model_generation = Node(FEATModel(), name = 'model_generation')
        run_level_analysis.connect(model_design, 'ev_files',  model_generation, 'ev_files')
        run_level_analysis.connect(model_design, 'fsf_files',  model_generation, 'fsf_files')

        # FILMGLS Node - Estimate first level model
        model_estimate = Node(FILMGLS(), name = 'model_estimate')
        model_estimate.inputs.smooth_autocorr = True
        model_estimate.inputs.mask_size = 5
        model_estimate.inputs.threshold = 1000
        run_level_analysis.connect(smoothing_func, 'out_file', model_estimate, 'in_file')
        run_level_analysis.connect(model_generation, 'con_file', model_estimate, 'tcon_file')
        run_level_analysis.connect(model_generation, 'design_file', model_estimate, 'design_file')

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        run_level_analysis.connect(
            model_estimate, 'results_dir', data_sink, 'run_level_analysis.@results')
        run_level_analysis.connect(
            model_generation, 'design_file', data_sink, 'run_level_analysis.@design_file')
        run_level_analysis.connect(
            model_generation, 'design_image', data_sink, 'run_level_analysis.@design_img')

        return run_level_analysis

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}')
        templates = [
                join(output_dir, 'results', 'cope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'tstat{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'varcope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'zstat{contrast_id}.nii.gz')
            ]
        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

    def get_subject_level_contrasts(subject_list: list, run_list: list):
        """ Return a list of contrasts and a dictionary of regressors
            for the subject level analysis """
        nb_subjects = len(subject_list)
        nb_runs = len(run_list)

        # Create a list of EVs (explanatory variables), one per subject
        ev_list = [f'ev{s}' for s in range(1,nb_subjects+1)]

        # Create a list of contrasts
        contrasts = [['', 'T', ev_list, [0.0] * len(ev_list)] for s in range(nb_subjects)]
        for index, _ in enumerate(contrasts):
            contrasts[index][3][index] = 1.0

        # Create a regressors dict
        regressors = {e:[0.0] * nb_subjects * nb_runs for e in ev_list}
        for index, key in enumerate(regressors.keys()):
            regressors[key][index * nb_runs:index * nb_runs + nb_runs] = [1.0] * nb_runs

        return contrasts, regressors

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
        - subject_level_analysis : nipype.WorkFlow
        """
        # Infosource Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'contrast_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

        # Datagrabber - Select copes and varcopes from the run level analysis
        select_copes = Node(DataGrabber(
            infields = ['contrast_id'],
            outfields = ['copes', 'varcopes']
            ), name = 'select_copes')
        select_copes.inputs.base_directory = join(
            self.directories.output_dir, 'run_level_analysis')
        select_copes.inputs.template = '*'
        select_copes.inputs.field_template = {
            'copes': join('_run_id_%s_subject_id_%s', 'results', 'cope%s.nii.gz'),
            'varcopes': join('_run_id_%s_subject_id_%s', 'results', 'varcope%s.nii.gz'),
            }
        select_copes.inputs.template_args = {
            'copes': [[r,s,'contrast_id'] for s,r in product(self.subject_list, self.run_list)],
            'varcopes': [[r,s,'contrast_id'] for s,r in product(self.subject_list, self.run_list)]
            }
        select_copes.inputs.sort_filelist = False
        subject_level.connect(information_source, 'contrast_id', select_copes, 'contrast_id')

        # Datagrabber - Select brain masks
        select_masks = Node(DataGrabber(
            outfields = ['masks']
            ), name = 'select_masks')
        select_masks.inputs.base_directory = join(
            self.directories.dataset_dir, 'derivatives', 'fmriprep')
        select_masks.inputs.template = '*_brainmask.nii.gz'
        select_masks.inputs.field_template = {
            'masks': join('sub-%s', 'func',
                'sub-%s_task-MGT_run-%s_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz')
            }
        select_masks.inputs.template_args = {
            'masks': [[s, s, r] for s, r in product(self.subject_list, self.run_list)]
            }
        select_masks.inputs.sort_filelist = False

        # Merge - Merge copes files for each subject
        merge_copes = Node(FSLMerge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'
        subject_level.connect(select_copes, 'copes', merge_copes, 'in_files')

        # Merge - Merge varcopes files for all runs for each subject
        merge_varcopes = Node(FSLMerge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'
        subject_level.connect(select_copes, 'varcopes', merge_varcopes, 'in_files')

        # Merge - Merge mask files for all runs for each subject
        merge_masks = Node(FSLMerge(), name = 'merge_masks')
        merge_masks.inputs.dimension = 't'
        subject_level.connect(select_masks, 'masks', merge_masks, 'in_files')

        # ImageMaths - Create a mask by selecting the min value along time axis
        mask_min = Node(ImageMaths(), name = 'mask_min')
        mask_min.inputs.op_string = '-Tmin'
        subject_level.connect(merge_masks, 'merged_file', mask_min, 'in_file')

        # ImageMaths - Mask copes
        mask_copes = Node(ImageMaths(), name = 'mask_copes')
        mask_copes.inputs.op_string = '-mas'
        subject_level.connect(merge_copes, 'merged_file', mask_copes, 'in_file')
        subject_level.connect(mask_min, 'out_file', mask_copes, 'in_file2')

        # ImageMaths - Mask copes
        mask_varcopes = Node(ImageMaths(), name = 'mask_varcopes')
        mask_varcopes.inputs.op_string = '-mas'
        subject_level.connect(merge_varcopes, 'merged_file', mask_varcopes, 'in_file')
        subject_level.connect(mask_min, 'out_file', mask_varcopes, 'in_file2')

        # Function - Create contrasts and regressors for second level
        get_contrasts = Node(Function(
            function = self.get_subject_level_contrasts,
            input_names = ['subject_list', 'run_list'],
            output_names = ['contrasts', 'regressors'],
            ), name='get_contrasts')
        get_contrasts.inputs.subject_list = self.subject_list
        get_contrasts.inputs.run_list = self.run_list

        # MultipleRegressDesign - Model subject level with multiple predictors (subjects)
        generate_model = Node(MultipleRegressDesign(), name = 'generate_model')
        subject_level.connect(get_contrasts, 'contrasts', generate_model, 'contrasts')
        subject_level.connect(get_contrasts, 'regressors', generate_model, 'regressors')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'fe'
        subject_level.connect(mask_min, 'out_file', estimate_model,  'mask_file')
        subject_level.connect(mask_copes, 'out_file', estimate_model, 'cope_file')
        subject_level.connect(mask_varcopes, 'out_file', estimate_model, 'var_cope_file')
        subject_level.connect(generate_model, 'design_mat', estimate_model, 'design_file')
        subject_level.connect(generate_model, 'design_con', estimate_model, 't_con_file')
        subject_level.connect(generate_model, 'design_grp', estimate_model, 'cov_split_file')

        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        subject_level.connect(
            mask_min, 'out_file', data_sink, 'subject_level_analysis.@mask')
        subject_level.connect(estimate_model, 'zstats', data_sink, 'subject_level_analysis.@stats')
        subject_level.connect(
            estimate_model, 'tstats', data_sink, 'subject_level_analysis.@tstats')
        subject_level.connect(estimate_model, 'copes', data_sink, 'subject_level_analysis.@copes')
        subject_level.connect(
            estimate_model, 'var_copes', data_sink, 'subject_level_analysis.@varcopes')

        return subject_level_analysis

    def get_subject_level_outputs(self):
        """ Return the names of the files the subject level analysis is supposed to generate. """

        parameters = {
            'contrast_id' : self.contrast_list,
            'subject_id' : self.subject_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir, 'subject_level_analysis',
            '_contrast_id_{contrast_id}_subject_id_{subject_id}')            
        templates = [
            join(output_dir, 'cope1.nii.gz'),
            join(output_dir, 'tstat1.nii.gz'),
            join(output_dir, 'varcope1.nii.gz'),
            join(output_dir, 'zstat1.nii.gz'),
            join(output_dir, 'sub-{subject_id}_task-MGT_run-01_bold_space-MNI152NLin2009cAsym_preproc_brain_mask_maths.nii.gz')
            ]

        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

    def get_one_sample_t_test_regressors(subject_list: list) -> dict:
        """
        Create dictionary of regressors for one sample t-test group analysis.

        Parameters:
            - subject_list: ids of subject in the group for which to do the analysis

        Returns:
            - dict containing named lists of regressors.
        """

        return dict(group_mean = [1 for _ in subject_list])

    def get_two_sample_t_test_regressors(
        equal_range_ids: list,
        equal_indifference_ids: list,
        subject_list: list,
        ) -> dict:
        """
        Create dictionary of regressors for two sample t-test group analysis.

        Parameters:
            - equal_range_ids: ids of subjects in equal range group
            - equal_indifference_ids: ids of subjects in equal indifference group
            - subject_list: ids of subject for which to do the analysis

        Returns:
            - regressors, dict: containing named lists of regressors.
            - groups, list: group identifiers to distinguish groups in FSL analysis.
        """

        # Create 2 lists containing n_sub values which are
        #  * 1 if the participant is on the group
        #  * 0 otherwise
        equal_range_regressors = [1 if i in equal_range_ids else 0 for i in subject_list]
        equal_indifference_regressors = [
            1 if i in equal_indifference_ids else 0 for i in subject_list
            ]

        # Create regressors output : a dict with the two list
        regressors = dict(
            equalRange = equal_range_regressors,
            equalIndifference = equal_indifference_regressors
        )

        # Create groups outputs : a list with 1 for equalRange subjects and 2 for equalIndifference
        groups = [1 if i == 1 else 2 for i in equal_range_regressors]

        return regressors, groups

    def get_group_level_analysis(self):
        """
        Return all workflows for the group level analysis.

        Returns;
            - a list of nipype.WorkFlow
        """

        methods = ['equalRange', 'equalIndifference', 'groupComp']
        return [self.get_group_level_analysis_sub_workflow(method) for method in methods]

    def get_group_level_analysis_sub_workflow(self, method):
        """
        Return a workflow for the group level analysis.

        Parameters:
            - method: one of 'equalRange', 'equalIndifference' or 'groupComp'

        Returns:
            - group_level_analysis: nipype.WorkFlow
        """
        # Infosource Node - iterate over the contrasts generated by the subject level analysis
        information_source = Node(IdentityInterface(
            fields = ['contrast_id']),
            name = 'information_source')
        information_source.iterables = [('contrast_id', self.contrast_list)]

        # SelectFiles Node - select necessary files
        templates = {
            'cope' : join(self.directories.output_dir,
                'subject_level_analysis', '_contrast_id_{contrast_id}_subject_id_*',
                'cope1.nii.gz'),
            'varcope' : join(self.directories.output_dir,
                'subject_level_analysis', '_contrast_id_{contrast_id}_subject_id_*',
                'varcope1.nii.gz'),
            'masks': join(self.directories.output_dir,
                'subject_level_analysis', '_contrast_id_1_subject_id_*',
                'sub-*_task-MGT_run-*_bold_space-MNI152NLin2009cAsym_preproc_brain_mask_maths.nii.gz')
            }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.results_dir

        # Datasink Node - save important files
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir

        # Function Node elements_in_string
        #   Get contrast of parameter estimates (cope) for these subjects
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_copes = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_copes', iterfield = 'input_str'
        )

        # Function Node elements_in_string
        #   Get variance of the estimated copes (varcope) for these subjects
        # Note : using a MapNode with elements_in_string requires using clean_list to remove
        #   None values from the out_list
        get_varcopes = MapNode(Function(
            function = elements_in_string,
            input_names = ['input_str', 'elements'],
            output_names = ['out_list']
            ),
            name = 'get_varcopes', iterfield = 'input_str'
        )

        # Merge Node - Merge cope files
        merge_copes = Node(FSLMerge(), name = 'merge_copes')
        merge_copes.inputs.dimension = 't'

        # Merge Node - Merge cope files
        merge_varcopes = Node(FSLMerge(), name = 'merge_varcopes')
        merge_varcopes.inputs.dimension = 't'

        # Split Node - Split mask list to serve them as inputs of the MultiImageMaths node.
        split_masks = Node(Split(), name = 'split_masks')
        split_masks.inputs.splits = [1, len(self.subject_list) - 1]
        split_masks.inputs.squeeze = True # Unfold one-element splits removing the list

        # MultiImageMaths Node - Create a subject mask by
        #   computing the intersection of all run masks.
        mask_intersection = Node(MultiImageMaths(), name = 'mask_intersection')
        mask_intersection.inputs.op_string = '-mul %s ' * (len(self.subject_list) - 1)

        # MultipleRegressDesign Node - Specify model
        specify_model = Node(MultipleRegressDesign(), name = 'specify_model')

        # FLAMEO Node - Estimate model
        estimate_model = Node(FLAMEO(), name = 'estimate_model')
        estimate_model.inputs.run_mode = 'flame1'

        # Randomise Node -
        randomise = Node(Randomise(), name = 'randomise')
        randomise.inputs.num_perm = 10000
        randomise.inputs.tfce = True
        randomise.inputs.vox_p_values = True
        randomise.inputs.c_thresh = 0.05
        randomise.inputs.tfce_E = 0.01

        # Compute the number of participants used to do the analysis
        nb_subjects = len(self.subject_list)

        # Declare the workflow
        group_level_analysis = Workflow(
            base_dir = self.directories.working_dir,
            name = f'group_level_analysis_{method}_nsub_{nb_subjects}')
        group_level_analysis.connect([
            (information_source, select_files, [('contrast_id', 'contrast_id')]),
            (select_files, get_copes, [('cope', 'input_str')]),
            (select_files, get_varcopes, [('varcope', 'input_str')]),
            (get_copes, merge_copes, [(('out_list', clean_list), 'in_files')]),
            (get_varcopes, merge_varcopes,[(('out_list', clean_list), 'in_files')]),
            (select_files, split_masks, [('masks', 'inlist')]),
            (split_masks, mask_intersection, [('out1', 'in_file')]),
            (split_masks, mask_intersection, [('out2', 'operand_files')]),
            (mask_intersection, estimate_model, [('out_file', 'mask_file')]),
            (mask_intersection, randomise, [('out_file', 'mask')]),
            (merge_copes, estimate_model, [('merged_file', 'cope_file')]),
            (merge_varcopes, estimate_model, [('merged_file', 'var_cope_file')]),
            (specify_model, estimate_model, [
                ('design_mat', 'design_file'),
                ('design_con', 't_con_file'),
                ('design_grp', 'cov_split_file')
                ]),
            (merge_copes, randomise, [('merged_file', 'in_file')]),
            (specify_model, randomise, [
                ('design_mat', 'design_mat'),
                ('design_con', 'tcon')
                ]),
            (randomise, data_sink, [
                ('t_corrected_p_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tcorpfile'),
                ('tstat_files', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstat')
                ]),
            (estimate_model, data_sink, [
                ('zstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@zstats'),
                ('tstats', f'group_level_analysis_{method}_nsub_{nb_subjects}.@tstats')
                ])
            ])

        if method in ('equalIndifference', 'equalRange'):

            # Setup a one sample t-test
            specify_model.inputs.contrasts = [
                ['group_mean', 'T', ['group_mean'], [1]],
                ['group_mean_neg', 'T', ['group_mean'], [-1]]
                ]

            # Function Node get_group_subjects - Get subjects in the group and in the subject_list
            get_group_subjects = Node(Function(
                function = list_intersection,
                input_names = ['list_1', 'list_2'],
                output_names = ['out_list']
                ),
                name = 'get_group_subjects'
            )
            get_group_subjects.inputs.list_1 = get_group(method)
            get_group_subjects.inputs.list_2 = self.subject_list

            # Function Node get_one_sample_t_test_regressors
            #   Get regressors in the equalRange and equalIndifference method case
            regressors_one_sample = Node(
                Function(
                    function = self.get_one_sample_t_test_regressors,
                    input_names = ['subject_list'],
                    output_names = ['regressors']
                ),
                name = 'regressors_one_sample',
            )

            # Add missing connections
            group_level_analysis.connect([
                (get_group_subjects, get_copes, [('out_list', 'elements')]),
                (get_group_subjects, get_varcopes, [('out_list', 'elements')]),
                (get_group_subjects, regressors_one_sample, [('out_list', 'subject_list')]),
                (regressors_one_sample, specify_model, [('regressors', 'regressors')])
            ])

        elif method == 'groupComp':

            # Select copes and varcopes corresponding to the selected subjects
            #   Indeed the SelectFiles node asks for all (*) subjects available
            get_copes.inputs.elements = self.subject_list
            get_varcopes.inputs.elements = self.subject_list

            # Setup a two sample t-test
            specify_model.inputs.contrasts = [
                ['equalRange_sup', 'T', ['equalRange', 'equalIndifference'], [1, -1]]
            ]

            # Function Node get_equal_range_subjects
            #   Get subjects in the equalRange group and in the subject_list
            get_equal_range_subjects = Node(Function(
                function = list_intersection,
                input_names = ['list_1', 'list_2'],
                output_names = ['out_list']
                ),
                name = 'get_equal_range_subjects'
            )
            get_equal_range_subjects.inputs.list_1 = get_group('equalRange')
            get_equal_range_subjects.inputs.list_2 = self.subject_list

            # Function Node get_equal_indifference_subjects
            #   Get subjects in the equalIndifference group and in the subject_list
            get_equal_indifference_subjects = Node(Function(
                function = list_intersection,
                input_names = ['list_1', 'list_2'],
                output_names = ['out_list']
                ),
                name = 'get_equal_indifference_subjects'
            )
            get_equal_indifference_subjects.inputs.list_1 = get_group('equalIndifference')
            get_equal_indifference_subjects.inputs.list_2 = self.subject_list

            # Function Node get_two_sample_t_test_regressors
            #   Get regressors in the groupComp method case
            regressors_two_sample = Node(
                Function(
                    function = self.get_two_sample_t_test_regressors,
                    input_names = [
                        'equal_range_ids',
                        'equal_indifference_ids',
                        'subject_list',
                    ],
                    output_names = ['regressors', 'groups']
                ),
                name = 'regressors_two_sample',
            )
            regressors_two_sample.inputs.subject_list = self.subject_list

            # Add missing connections
            group_level_analysis.connect([
                (get_equal_range_subjects, regressors_two_sample, [
                    ('out_list', 'equal_range_ids')
                    ]),
                (get_equal_indifference_subjects, regressors_two_sample, [
                    ('out_list', 'equal_indifference_ids')
                    ]),
                (regressors_two_sample, specify_model, [
                    ('regressors', 'regressors'),
                    ('groups', 'groups')])
            ])

        return group_level_analysis

    def get_group_level_outputs(self):
        """ Return all names for the files the group level analysis is supposed to generate. """

        # Handle equalRange and equalIndifference
        parameters = {
            'contrast_id': self.contrast_list,
            'method': ['equalRange', 'equalIndifference'],
            'file': [
                'randomise_tfce_corrp_tstat1.nii.gz',
                'randomise_tfce_corrp_tstat2.nii.gz',
                'randomise_tstat1.nii.gz',
                'randomise_tstat2.nii.gz',
                'tstat1.nii.gz',
                'tstat2.nii.gz',
                'zstat1.nii.gz',
                'zstat2.nii.gz'
                ],
            'nb_subjects' : [str(len(self.subject_list))]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            'group_level_analysis_{method}_nsub_{nb_subjects}',
            '_contrast_id_{contrast_id}',
            '{file}'
            )

        return_list = [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        # Handle groupComp
        parameters = {
            'contrast_id': self.contrast_list,
            'file': [
                'randomise_tfce_corrp_tstat1.nii.gz',
                'randomise_tstat1.nii.gz',
                'zstat1.nii.gz',
                'tstat1.nii.gz'
            ]
        }
        parameter_sets = product(*parameters.values())
        template = join(
            self.directories.output_dir,
            f'group_level_analysis_groupComp_nsub_{len(self.subject_list)}',
            '_contrast_id_{contrast_id}', '{file}')
        return_list += [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return all hypotheses output file names. """

        nb_sub = len(self.subject_list)
        files = [
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_1', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat2.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat2.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat2.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalIndifference_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_equalRange_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'randomise_tfce_corrp_tstat1.nii.gz'),
            join(f'group_level_analysis_groupComp_nsub_{nb_sub}',
                '_contrast_id_2', 'zstat1.nii.gz')
        ]
        return [join(self.directories.output_dir, f) for f in files]
