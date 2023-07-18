#!/usr/bin python3
# -*- coding: utf-8 -*-

"""
UE_interface module - import into 3rd party code to provide the functionality for the
generation of metrics for Unlearning Effectiveness.

Recommended imports for a new code base:

1. Imports:

    from UE_interface import (
        ue_setup,
        ue_start_training,
        ue_stop_training,
        ue_start_unlearning,
        ue_stop_unlearning,
        ue_print_run_stats
)

2. Additional arguments:

2.1 Mandatory arguments

    parser.add_argument("-ue_sm", '--UE_store_model_filename', default=None, type=str,
                        help="UE - Save the learnt model to the named file")
    parser.add_argument("-ue_nt", '--UE_nametag', default=None, type=str,
                        help="UE - Nametag to use for stats")

2.2 Optional arguments
    Dependent on 3rd party code - whatever needs to be parameterised.

3. Embedded code:

3.1 Before start of any UE operation - create and initialise UE storage class :

    UE = ue_setup("Description", UE_nametag, device, dataset_name)
    Args:
        Description - Free text description of the 3rd party code
        UE_nametag - tag used to consistently store stats
        device - CUDA status - cpu or gpu
        dataset_name - name of dataset - e.g. CIFAR10

    Returns
        UE - class instance of UE helper storage class

3.2 Before training:

    ue_start_training(UE, training_data_size, test_data_size))
    Args:
        UE - UE helper class instance
        training_data_size - length of training data
        test_data_size - length of test data

3.3 After training:

     ue_stop_training(UE, training_loss_score)
     Args:
         UE - UE helper class instance
         training_loss_score - loss score after training

    ue_store_model(model, UE_nametag, UE_store_model_filename)
    Args:
        model - torch model
        UE_nametag -tag name used for current run
        UE_store_model_filename - Filename to use for model storage


3.4 Before unlearning:

    ue_start_unlearning(UE, number_of_items_to_remove)
    Args:
        UE - UE helper class instance
        number_of_items_to_remove - removal count

3.5 After unlearning:

    ue_stop_unlearning(UE, unlearn_loss_score)
    Args:
        UE - UE helper class instance
        unlearn_loss_score - loss score after unlearning

    ue_store_model(model, UE_nametag, UE_store_model_filename)
    Args:
        model - torch model
        UE_nametag -tag name used for current run
        UE_store_model_filename - Filename to use for model storage

3.6 At end of all train/unlearn/inference activity:

    ue_print_run_stats(UE)
    Print all UE stats to stdout for collection by wrapper process.
    Args:
        UE - UE helper class instance
"""

from datetime import datetime
import json
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import psutil
from subprocess import Popen, PIPE
import sys
import time
import torch

UE_KEY_CUDA = "UE_cuda_status"
UE_KEY_DATASET = "UE_dataset"
UE_KEY_TRAINING_DATA_SIZE = "UE_training_data_size"
UE_KEY_TEST_DATA_SIZE = "UE_test_data_size"
UE_KEY_TRAINING_START = "UE_training_start"
UE_KEY_TRAINING_END = "UE_training_end"
UE_KEY_TRAINING_TIME = "UE_training_time"
UE_KEY_TRAINING_LOSS_SCORE = "UE_training_loss_score"
UE_KEY_UNLEARN_DATA_SIZE = "UE_unlearn_data_size"
UE_KEY_UNLEARN_START = "UE_unlearn_start"
UE_KEY_UNLEARN_END = "UE_unlearn_end"
UE_KEY_UNLEARN_TIME = "UE_unlearn_time"
UE_KEY_UNLEARN_LOSS_SCORE = "UE_unlearn_loss_score"
UE_KEY_INFERENCE = "inference"

UE_ERROR_LOGGER = "UE_error_logger"

UE_TRAIN_MODEL = 'train'
UE_UNLEARN_MODEL = 'unlearn'
UE_TRAIN_UNLEARN = 'train_and_unlearn'
UE_VALID_MODES = [
    UE_TRAIN_MODEL,
    UE_UNLEARN_MODEL
]

HOME_DIR = os.path.expanduser('~')
UE_STATS_STORE_DIRECTORY = HOME_DIR + '/unlearning_effectiveness/stats'
UE_MODEL_STORE_DIRECTORY = HOME_DIR + '/unlearning_effectiveness/models'
UE_SCRATCH_STORE_DIRECTORY = HOME_DIR + '/unlearning_effectiveness/scratch'

STOREFILE_SUFFIX = "ue_store.csv"
UE_OPERATION_TRAIN = 'train'
UE_OPERATION_TRAIN_UNLEARN = 'train_unlearn'
UE_OPERATION_UNLEARN = 'unlearn'
UE_OPERATION_INFERENCE = 'inference'
UE_OPERATION_ALL = "train_unlearn_inference"
UE_OPERATION_WATERMARK = "watermark"
UE_OPERATION_DISPLAY_TAGS = 'tags'
UE_OPERATION_DISPLAY_STATS = 'stats'
UE_VALID_OPERATIONS = [
    UE_OPERATION_TRAIN,
    UE_OPERATION_UNLEARN,
    UE_OPERATION_TRAIN_UNLEARN,
    UE_OPERATION_INFERENCE,
    UE_OPERATION_WATERMARK,
    UE_OPERATION_DISPLAY_TAGS,
    UE_OPERATION_DISPLAY_STATS,
    UE_OPERATION_ALL
]

UE_VALID_WRITE_OPERATIONS = [
    UE_OPERATION_TRAIN,
    UE_OPERATION_UNLEARN,
    UE_OPERATION_TRAIN_UNLEARN,
    UE_OPERATION_ALL
]
UE_VALID_READ_OPERATIONS = [
    UE_OPERATION_TRAIN,
    UE_OPERATION_UNLEARN,
    UE_OPERATION_ALL
]

UE_STATS_INTERVAL_SECS = 5
UE_GPU_STATS = 'gpu'
UE_CPU_STATS = 'cpu'

DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
UNIX_EPOCH = datetime.strptime('1970-01-01_00:00:00', DATETIME_FORMAT)
TIMEDELTA_FORMAT = '%H:%M:%S.%f'


class UEHelper(object):
    """
    Class to be instantiated by third party code to store UE operation results.
    """

    VALID_UE_KEYWORDS = [UE_KEY_CUDA,
                         UE_KEY_DATASET,
                         UE_KEY_TRAINING_DATA_SIZE,
                         UE_KEY_TEST_DATA_SIZE,
                         UE_KEY_TRAINING_START,
                         UE_KEY_TRAINING_END,
                         UE_KEY_TRAINING_LOSS_SCORE,
                         UE_KEY_UNLEARN_DATA_SIZE,
                         UE_KEY_UNLEARN_START,
                         UE_KEY_UNLEARN_END,
                         UE_KEY_UNLEARN_TIME,
                         UE_KEY_UNLEARN_LOSS_SCORE
                         ]

    CUDA_STATE_CPU = 'cpu'
    CUDA_STATE_GPU = 'cuda'
    VALID_CUDA_STATES = [CUDA_STATE_CPU, CUDA_STATE_GPU]

    def __init__(self, title, nametag):
        self.title = title
        self.nametag = nametag
        self.cuda_status = 'cpu'
        self.dataset = None

        # Training metrics
        self.training_data_size = 0
        self.test_data_size = 0
        self.training_loss_score = 0.0
        self.training_start = None
        self.training_time = None

        # Unlearning metrics
        self.unlearn_data_size = 0
        self.unlearn_loss_score = 0.0
        self.unlearn_start = None
        self.unlearn_time = None

    def get_nametag(self):
        return self.nametag

    def set_ue_value(self, keyword, value):
        if keyword == UE_KEY_CUDA:
            if value:
                self.cuda_status = self.CUDA_STATE_GPU
            else:
                self.cuda_status = self.CUDA_STATE_CPU
            return
        elif keyword == UE_KEY_DATASET:
            if not isinstance(value, str):
                print(f"Dataset must be a string. Got '{value}")
                return
            self.dataset = value
        elif keyword == UE_KEY_TRAINING_DATA_SIZE:
            if not isinstance(value, int):
                print(f"Training data size must be an integer. Got '{value}")
                return
            self.training_data_size = value
        elif keyword == UE_KEY_TEST_DATA_SIZE:
            if not isinstance(value, int):
                print(f"Test data size must be an integer. Got '{value}")
                return
            self.test_data_size = value
        elif keyword == UE_KEY_TRAINING_START:
            self.training_start = datetime.utcnow()
        elif keyword == UE_KEY_TRAINING_END:
            if self.training_start is None:
                print("Training has not been started")
                return
            self.training_time = datetime.utcnow() - self.training_start
            self.training_start = None
        elif keyword == UE_KEY_TRAINING_LOSS_SCORE:
            self.training_loss_score = value
        elif keyword == UE_KEY_UNLEARN_DATA_SIZE:
            if not isinstance(value, int):
                print(f"Training data size must be an integer. Got '{value}")
                return
            self.unlearn_data_size = value
        elif keyword == UE_KEY_UNLEARN_START:
            self.unlearn_start = datetime.utcnow()
        elif keyword == UE_KEY_UNLEARN_END:
            if self.unlearn_start is None:
                print("Training has not been started")
                return
            self.unlearn_time = datetime.utcnow() - self.unlearn_start
            self.unlearn_start = None
        elif keyword == UE_KEY_UNLEARN_LOSS_SCORE:
            self.unlearn_loss_score = value
        else:
            print(f"Unknown keyword {keyword}")
        return

    def get_training_time(self):
        return self.training_time

    def get_unlearn_time(self):
        return self.unlearn_time

    def print_ue_training_values(self):
        """
        print current UE training values as key=value pairs to pass to the wrapper
        """
        print(f"{UE_KEY_CUDA}={self.cuda_status}")
        print(f"{UE_KEY_DATASET}={self.dataset}")
        print(f"{UE_KEY_TRAINING_DATA_SIZE}={self.training_data_size}")
        print(f"{UE_KEY_TEST_DATA_SIZE}={self.test_data_size}")
        print(f"{UE_KEY_TRAINING_TIME}={self.training_time}")
        print(f"{UE_KEY_TRAINING_LOSS_SCORE}={self.training_loss_score}")

    def print_ue_unlearn_values(self):
        """
        print current UE unlearning values as key=value pairs to pass to the wrapper
        """
        print(f"{UE_KEY_CUDA}={self.cuda_status}")
        print(f"{UE_KEY_DATASET}={self.dataset}")
        print(f"{UE_KEY_UNLEARN_DATA_SIZE}={self.unlearn_data_size}")
        print(f"{UE_KEY_UNLEARN_TIME}={self.unlearn_time}")
        print(f"{UE_KEY_UNLEARN_LOSS_SCORE}={self.unlearn_loss_score}")


def ue_print_formatted_bytestring(header, message):
    """
    Writes message to stdout, splitting by linebreaks. Used for passing errors to the UE wrapper.
    Args:
        header (string): text to display before output
        message (bytes): byte string for display
    """
    print(f"{header}\n")
    message = message.decode()
    lines = message.split('\n')
    for line in lines:
        print(line)


def ue_get_files_in_directory(path):
    """
    Get list of files in the specified directory path.
    Side effect: Creates the directory if it does not exist.
    Args:
        path (string): directory path.
    Returns:
        (list): List of raw filenames in path. Path is not appended to the names
    """
    if not os.path.exists(UE_MODEL_STORE_DIRECTORY):
        os.makedirs(UE_MODEL_STORE_DIRECTORY)
    return [f for f in listdir(path) if isfile(join(path, f))]


def get_ratio(top, bottom):
    """
    Get the ratio of top to bottom with divide by zero protection
    Args:
        top (string): numerator
        bottom (string): denominator
    Returns:
        (float): ratio of top to bottom
    """
    return 1 if float(bottom) == 0.0 else float(top) / float(bottom)


def zero_pad_timedelta(timedelta_string):
    """
    Zero-pad a timedelta hours field (if required)
    Args:
        timedelta_string (string): timedelta in string format
    Returns:
        (string) zero padded string
    """
    if timedelta_string.startswith('0:'):
        return f"0{timedelta_string}"
    else:
        return timedelta_string


def ue_store_model(model_state, nametag, filename, train_or_unlearn=UE_TRAIN_MODEL):
    """
    Store the model to the named file.
    Modify the filename to include _unlearn if this is an unlearn operation.
    Args:
        model_state (DICT): Pytorch model state in JSON format
        nametag (string): tag name for the scenario
        filename (string): Full filename including path
        train_or_unlearn (string): is this a trained or unlearnt model.
    """
    if filename is not None:
        if train_or_unlearn == UE_UNLEARN_MODEL:
            filename_split = filename.split('.')
            filename = filename_split[0] + '_unlearn' + '.pt'
        torch.save(model_state, filename)
        print(f"Model state tagged as {nametag} saved to {filename}")


def ue_load_trained_model(nametag):
    """
    Loads a trained PyTorch model using the specified nametag.
    Args:
        nametag (string): Tag name for the model
    returns:
        (model): Trained pytorch model
    """
    if not os.path.exists(UE_MODEL_STORE_DIRECTORY):
        print("No models exist. Please train one before loading")
        os.makedirs(UE_MODEL_STORE_DIRECTORY)
        return False, None
    existing_models = ue_get_files_in_directory(UE_MODEL_STORE_DIRECTORY)
    if len(existing_models) == 0:
        print("No models exist. Please train one before loading")
        return False, None
    requested_model = f"{nametag}.pt"
    if requested_model not in existing_models:
        print(f"Requested model {requested_model} does not exist in directory {UE_MODEL_STORE_DIRECTORY}")
        return False, None
    model_filename = f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}"
    model = torch.load(model_filename)
    return True, model

def ue_get_unlearn_data(unlearn_filename):
    """
    Reads the contents of the unlearn_filename and returns an iterator to the data for unlearning.
    Args:
        unlearn_filename (string): Name of file containing data to unlearn
    Return:
        (iterator) data to be unlearnt
    """
    try:
        unlearn_pd = pd.read_csv(unlearn_filename)
    except Exception as exp:
        print(f"Specified unlearning data file load issue, error is {exp}")
        sys.exit(1)
    # TODO


def ue_store_metrics(nametag,
                     operation,
                     cuda_status,
                     dataset,
                     epochs,
                     training_size,
                     test_size,
                     unlearning_size,
                     operation_score,
                     operation_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB
                     ):
    """
    Store the score and training time for the operation. Appends to a CSV results file
    named for the nametag, creating it if it doesn't exist. Each run has a datestamp to make each unique
    and logged separately.
    Args:
        nametag (string): tag name for the model
        operation (string): TRAINING or UNLEARNING or INFERENCE
        cuda_status (string): True if CUDA is enabled for the operation
        dataset (string): dataset in use.
        epochs (int): Number of epochs used in learning
        training_size (int): Number of elements in dataset used for training
        test_size (int): Number of elements in dataset used for testing
        unlearning_size (int): Number of elements in unlearning data set
        operation_score (float): score for the named operation
        operation_time (timedelta): time taken to perform the operation
        cpu_cumulative_seconds (float): Total CPU seconds used in training
        cpu_average_memory_MiB (float): Average system memory usage during training
        cpu_peak_memory_MiB (float): Peak system memory usage during training
        gpu_cumulative_seconds (float): Total GPU seconds used in training,
        gpu_average_memory_MiB (float): Average GPU memory usage during training
        gpu_peak_memory_MiB (float): peak memory usage during training
:return
        -
    """
    if operation not in UE_VALID_WRITE_OPERATIONS:
        print(f"store_metrics: invalid operation {operation}, must be one of {UE_VALID_WRITE_OPERATIONS}")
        return
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        os.makedirs(UE_STATS_STORE_DIRECTORY)
    store_file = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_{STOREFILE_SUFFIX}"
    timestamp = datetime.strftime(datetime.utcnow(), DATETIME_FORMAT)
    store_csv_data = f"{nametag}," \
                     f"{timestamp}," \
                     f"{operation}," \
                     f"{cuda_status}," \
                     f"{epochs}," \
                     f"{dataset}," \
                     f"{training_size}," \
                     f"{test_size}," \
                     f"{unlearning_size}," \
                     f"{operation_score}," \
                     f"{operation_time}," \
                     f"{cpu_cumulative_seconds}," \
                     f"{cpu_average_memory_MiB}," \
                     f"{cpu_peak_memory_MiB}," \
                     f"{gpu_cumulative_seconds}," \
                     f"{gpu_average_memory_MiB}," \
                     f"{gpu_peak_memory_MiB}," \
                     f"\n"
    if not os.path.exists(store_file):
        store_csv_header = f"nametag," \
                           f"dateTime," \
                           f"operation," \
                           f"CUDA," \
                           f"epochs," \
                           f"dataset," \
                           f"trainSz," \
                           f"testSz," \
                           f"unlearnSz," \
                           f"score," \
                           f"time," \
                           f"CPUSecs," \
                           f"CPUMemAvg," \
                           f"CPUMemPeak," \
                           f"GPUSecs," \
                           f"GPUMemAvg," \
                           f"GPUMemPeak," \
                           f"\n"
        store_csv = store_csv_header + store_csv_data
        with open(store_file, "w") as fd:
            fd.write(store_csv)
    else:
        with open(store_file, "a") as fd:
            fd.write(store_csv_data)
    print(f'Stored metrics for {operation} tag: "{nametag}" to "{store_file}"')


def ue_get_stored_nametags(display=False):
    """
    Gets a list of nametags in UE_STATS_STORE_DIRECTORY.
    Displays the list if display is True
    Returns the list and the name (with paths) of the files in the UE_STATS_STORE_DIRECTORY if display is False

    Args:
        display (bool): Display nametags if True
    Returns:
         Nothing if display is False, otherwise:
        (list): List of nametags
        (list): List of filenames
    """
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        return []
    filename_list = [f for f in listdir(UE_STATS_STORE_DIRECTORY) if isfile(join(UE_STATS_STORE_DIRECTORY, f))]
    nametag_list = []
    for filename in filename_list:
        nametag = filename.split('_')[0]
        nametag_list.append(nametag)

    if display:
        print("Current list of nametags in UE stats storage area:")
        for tag in nametag_list:
            print(tag)
        return

    return nametag_list, filename_list



def ue_get_effectiveness_stats(nametag):
    """
    Returns the latest stats for a nametag model as a JSON dict
    Where multiple stats exist for a nametag model, return the latest for each category.
    Args:
        nametag (string): tag name for the model
    Returns:
        (DICT) {
                train: {
                    train_elapsed - (datetime) training elapsed time
                    train_cpu_seconds (float) CPU seconds used during training
                    train_cpu_avg_mem (float) average CPU memory usage during training
                    train_gpu_seconds (float) GPU seconds used during training
                    train_gpu_avg_mem (float) GPU average memory usage during training
                    train_accuracy (float) training accuracy score
                    }
                unlearn: {
                    unlearn_elapsed - (datetime) training elapsed time
                    unlearn_cpu_seconds (float) CPU seconds used during training
                    unlearn_cpu_avg_mem (float) average CPU memory usage during training
                    unlearn_gpu_seconds (float) GPU seconds used during training
                    unlearn_gpu_avg_mem (float) GPU average memory usage during training
                    unlearn_accuracy (float) training accuracy score
                    }
                inference: {
                    inference_score (float) membership inference score.
                    }
                }
    """
    stored_nametag_list, filename_list = ue_get_stored_nametags()
    if nametag not in stored_nametag_list:
        print(f"No metrics for Nametag {nametag} exist in the model store")
        return

    train = {}
    unlearn = {}
    train_inference = {}
    unlearn_inference = {}
    last_train = UNIX_EPOCH
    last_unlearn = UNIX_EPOCH
    last_train_inference = UNIX_EPOCH
    last_unlearn_inference = UNIX_EPOCH

    for filename in filename_list:
        filename_nametag = filename.split('_')[0]
        if filename_nametag == nametag:
            full_filename = f"{UE_STATS_STORE_DIRECTORY}/{filename}"
            df_stats = pd.read_csv(full_filename, index_col=False)
            for i, row in df_stats.iterrows():
                if row['operation'] == 'train':
                    operation_time = datetime.strptime(row['dateTime'], DATETIME_FORMAT)
                    if operation_time > last_train:
                        train = {
                            'elapsed': row['time'],
                            'cpu_seconds': row['CPUSecs'],
                            'cpu_avg_mem': row['CPUMemAvg'],
                            'gpu_seconds': row['GPUSecs'],
                            'gpu_avg_mem': row['GPUMemAvg'],
                            'accuracy': row['score']
                        }
                        last_train = operation_time
                elif row['operation'] == 'unlearn':
                    operation_time = datetime.strptime(row['dateTime'], DATETIME_FORMAT)
                    if operation_time > last_unlearn:
                        unlearn = {
                            'elapsed': row['time'],
                            'cpu_seconds': row['CPUSecs'],
                            'cpu_avg_mem': row['CPUMemAvg'],
                            'gpu_seconds': row['GPUSecs'],
                            'gpu_avg_mem': row['GPUMemAvg'],
                            'accuracy': row['score']
                        }
                        last_unlearn = operation_time
                elif row['operation'] == 'train_inference':
                    operation_time = datetime.strptime(row['dateTime'], DATETIME_FORMAT)
                    if operation_time > last_train_inference:
                        train_inference = {
                            'score': row['InferenceScore']
                        }
                        last_train_inference = operation_time
                elif row['operation'] == 'unlearn_inference':
                    operation_time = datetime.strptime(row['dateTime'], DATETIME_FORMAT)
                    if operation_time > last_unlearn_inference:
                        unlearn_inference = {
                            'score': row['InferenceScore']
                        }
                        last_unlearn_inference = operation_time
    return_dict = {
        'train': train,
        'unlearn': unlearn,
        'train_inference': train_inference,
        'unlearn_inference': unlearn_inference
    }
    return return_dict


def ue_display_stats_and_generate_mue(nametag, stats):
    """
    Format and display run state stats and calculate their Measurement of Unlearning Effectiveness (MUE) value
    MUE is generated as the product of:
        RR - Resource Ratio - between unlearning and training for all 5 resource metrics
        LR - Loss Ratio - between unlearning and training loss scores.
        IR - Inference ratio - between unlearning and training membership inference scores
    Args:
        nametag (string): tag name for the model
        stats (DICT): stats information
    """

    # Display the stats
    print(f"Training scores: for nametag {nametag}:")
    print(json.dumps(stats, indent=4))
    if len(stats['train']) == 0 or len(stats['unlearn']) == 0:
        print("Not enough stats to generate MUE. Needs both training and unlearning stats to work ")
        return


    # Generate the MUE
    unlearn_time = zero_pad_timedelta(stats['unlearn']['elapsed'])
    train_time = zero_pad_timedelta(stats['train']['elapsed'])
    unlearn_seconds = datetime.strptime(unlearn_time, TIMEDELTA_FORMAT).timestamp()
    train_seconds = datetime.strptime(train_time, TIMEDELTA_FORMAT).timestamp()
    train_ratio = unlearn_seconds / train_seconds
    cpu_ratio = get_ratio(stats['unlearn']['cpu_seconds'], stats['train']['cpu_seconds'])
    cpu_mem_ratio = get_ratio(stats['unlearn']['cpu_avg_mem'], stats['train']['cpu_avg_mem'])
    gpu_ratio = get_ratio(stats['unlearn']['gpu_seconds'], stats['train']['gpu_seconds'])
    gpu_mem_ratio = get_ratio(stats['unlearn']['gpu_avg_mem'], stats['train']['gpu_avg_mem'])
    accuracy_ratio = get_ratio(stats['unlearn']['accuracy'], stats['train']['accuracy'])

    if len(stats['train_inference']) == 0 or len(stats['unlearn_inference']) == 0:
        # Set this to 1 - assume unlearning is successful but mark the score as being incomplete
        inference_ratio = 1.0
        inference_flag = " (No Membership Inference data available - result is incomplete)"
    else:
        inference_ratio = stats['unlearn_inference']['score'] / stats['train_inference']['score']
        inference_flag = ""

    MUE_score = train_ratio * cpu_ratio * cpu_mem_ratio * gpu_ratio * gpu_mem_ratio * accuracy_ratio * inference_ratio

    print(f"MUE score for nametag {nametag} is {MUE_score}{inference_flag}")

def ue_extract_token_value(message, token):
    """
    Parses message for specified token and returns the value associated with the token
    Returns after first occurrence of token in message
    Args:
        message (string): spring to search for token in
        token (string): search string
    """
    if '\\n' in message:
        lines = str(message).split('\\n')
    else:
        if '\n' in message:
            lines = str(message).split('\n')
        else:
            lines = message
    for line in lines:
        if token in line:
            return line.split('=')[1]
    return None

def ue_log_error(message):
    """
    Log error message to be interpreted by the wrapper.
    """
    print(f"{UE_ERROR_LOGGER}={str(message)}")

def ue_set_stats_mode_train(nametag, verbose=False):
    """
    Sets the mode for the stats updates for the nametag to TRAIN
    Args:
        nametag (string): tag name for the model
        verbose (bool): verbose mode.
    """
    if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
        os.mkdir(UE_SCRATCH_STORE_DIRECTORY)
    UE_STATS_MODE_FILE = UE_SCRATCH_STORE_DIRECTORY + f"/{nametag}-mode.dat"
    with open(UE_STATS_MODE_FILE, 'w') as mode_file:
        mode_file.write(UE_TRAIN_MODEL)
    if verbose:
        print(f"Stats mode updated to '{UE_TRAIN_MODEL}'")

def ue_set_stats_mode_unlearn(nametag, verbose=False):
    """
    Sets the mode for the stats updates for the nametag to UNLEARN
    Args:
        nametag (string): tag name for the model
        verbose (bool): verbose mode.
    """
    if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
        os.mkdir(UE_SCRATCH_STORE_DIRECTORY)
    UE_STATS_MODE_FILE = UE_SCRATCH_STORE_DIRECTORY + f"/{nametag}-mode.dat"
    with open(UE_STATS_MODE_FILE, 'w') as mode_file:
        mode_file.write(UE_UNLEARN_MODEL)
    if verbose:
        print(f"Stats mode updated to '{UE_TRAIN_MODEL}'")


def ue_get_stats_mode(nametag, verbose=False):
    """
    Gets the current mode for the stats updates for the current instance. This is gleaned
    from a file in the stats store directory called <nametag>-mode.dat.
    Args:
        nametag (string): tag name for the model
        verbose (bool): verbose mode.
    Return:
        (string): current mode.
    """
    # Assuming training
    mode = UE_TRAIN_MODEL
    if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
        ue_set_stats_mode_train(nametag)
    else:
        ue_stats_mode_file = UE_SCRATCH_STORE_DIRECTORY + f"/{nametag}-mode.dat"
        if not os.path.exists(ue_stats_mode_file):
            ue_set_stats_mode_train(nametag)
        else:
            with open(ue_stats_mode_file, 'r') as mode_file:
                mode = mode_file.readline()
    if mode not in UE_VALID_MODES:
        print(f"ue_get_stats_mode read bad mode {mode}")
        sys.exit(1)
    if verbose:
        print(f"Stats mode is '{mode}'")
    return mode


def ue_get_and_store_gpu_stats(nametag, interval, verbose):
    """
    Retrieve GPU run-time stats and store them in a stats file.
    Stores the % CGU utilization and memory usage (in MiB) every STATS_INTERVAL seconds.
    This runs as a thread in parallel with a Training/unlearning operation
    gets terminated by the operation when it completes.
    Args:
        nametag (string): tag name for the model
        interval (int): Number of seconds between readings
        verbose (bool): verbose mode.
    """
    system_command =\
        ["nvidia-smi",
         f"--query-gpu=timestamp,utilization.gpu,memory.used",
         f"--format=csv,noheader"]

    header = 'mode,' \
             'timestamp,' \
             'processor%,' \
             'memoryMiB\n'
    if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
        os.makedirs(UE_SCRATCH_STORE_DIRECTORY)
    filename = f"{UE_SCRATCH_STORE_DIRECTORY}/{nametag}_gpu.csv"
    with open(filename, 'w') as stats:
        stats.write(header)
    if verbose:
        print(f"Getting GPU stats using '{system_command}'")
    while True:
        process = Popen(system_command, stdout=PIPE, stderr=PIPE)
        gpu_stats, stderr = process.communicate()
        if len(stderr) != 0:
            ue_print_formatted_bytestring("Error message:", stderr)
        else:
            if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
                os.mkdir(UE_SCRATCH_STORE_DIRECTORY)
            mode = ue_get_stats_mode(nametag, verbose)
            filename = f"{UE_SCRATCH_STORE_DIRECTORY}/{nametag}_gpu.csv"
            output_line = f"{mode},{gpu_stats.decode('utf-8').replace(', ', ',').replace(' %', '').replace(' MiB', '')}"
            with open(filename, 'a') as stats:
                stats.write(output_line)
        time.sleep(interval)


def ue_get_and_store_system_stats(nametag, interval, verbose):
    """
    Get and store the current runtime stats and store in a stats file.
    This runs as a thread in parallel with a Training/unlearning operation
    gets terminated by the operation when it completes.
    Args:
        nametag (string): tag name for the model
        interval (int): Number of seconds between readings
        verbose (bool): verbose mode.
    """
    header = 'mode,' \
             'timestamp,' \
             'processor%,' \
             'memoryMiB\n'
    if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
        os.makedirs(UE_SCRATCH_STORE_DIRECTORY)
    filename = f"{UE_SCRATCH_STORE_DIRECTORY}/{nametag}_cpu.csv"
    with open(filename, 'w') as stats:
        stats.write(header)
    if verbose:
        print(f"Getting CPU stats")
    while True:
        timestamp = datetime.utcnow().strftime(DATETIME_FORMAT)
        cpu_pct = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory[2]
        if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
            os.mkdir(UE_SCRATCH_STORE_DIRECTORY)
        mode = ue_get_stats_mode(nametag, verbose)
        filename = f"{UE_SCRATCH_STORE_DIRECTORY}/{nametag}_cpu.csv"
        output_line = f"{mode},{timestamp},{cpu_pct},{memory_percent}\n"
        with open(filename, 'a') as stats:
            stats.write(output_line)
        time.sleep(interval)


def ue_get_gpu_cpu_stats(nametag, stats_type, operation, verbose):
    """
    Reads the stats file for the named stats type and summarises its contents to generate
    cumulative CPG/GPU, average memory and peak memory values for a run
    Args:
        nametag (string): tag name for the model
        stats_type (string): CPU or GPU
        operation (string): train or unlearn operation
        verbose (bool): verbose mode.
    returns
       cumulative CPU/GPU seconds
       Average memory usage (MB)
       Peak memory usage (MB)
    """
    if not os.path.exists(UE_SCRATCH_STORE_DIRECTORY):
        os.makedirs(UE_SCRATCH_STORE_DIRECTORY)
    if stats_type == UE_CPU_STATS:
        filename = f"{UE_SCRATCH_STORE_DIRECTORY}/{nametag}_cpu.csv"
    elif stats_type == UE_GPU_STATS:
        filename = f"{UE_SCRATCH_STORE_DIRECTORY}/{nametag}_gpu.csv"
    else:
        print(f"Unknown stats type {stats_type}")
        sys.exit(1)
    if not os.path.exists(filename):
        print(f"Model has no stored stats in {filename}")
        sys.exit(1)
    if verbose:
        print(f"Loading stats for nametag {nametag} {stats_type} data from '{filename}'")
    df_stats = pd.read_csv(filename, index_col=None)
    # Filter out stats not of the requested mode.
    df_stats = df_stats[df_stats['mode'] == operation]
    count = len(df_stats)

    # Prevent divide by zero error on failure.
    if count == 0:
        return 0, 0, 0

    # The processor % is the % of CPU/GPU time used in the last STATS_INTERVAL seconds.
    # The number of seconds of CPU time in this interval is calculated as below
    cumulative_processor_seconds = (int(df_stats['processor%'].sum()) * UE_STATS_INTERVAL_SECS) / 100
    avg_memory_MiB = round(int(df_stats['memoryMiB'].sum()) / count, 2)
    peak_memory_MiB = int(df_stats['memoryMiB'].max())
    if verbose:
        print(f"get_stats: {nametag}: ")
    return cumulative_processor_seconds, avg_memory_MiB, peak_memory_MiB



"""
Main interface methods
"""
def ue_setup(title, nametag, cuda, dataset):
    """
    Instantiate the Unlearning Effectiveness helper with initial parameters
    Args:
        title (string): Friendly name for the class instantiation
        nametag (string): tag name for the model
        cuda (bool): CUDA status
        dataset (string): Name of dataset being used for this instantiation
    Return:
        (class): Instance of the UDHelper class.
    """
    UE = UEHelper(title, nametag)
    UE.set_ue_value(UE_KEY_CUDA, cuda)
    UE.set_ue_value(UE_KEY_DATASET, dataset)
    return UE

def ue_start_training(UE, train_size, test_size, verbose=False):
    """
    Start a UE training session after storing parameters in class variables and
    starting a timer to log the duration of the run.
    Args:
        UE (class): instance of the UEHelper class
        train_size (int): Number of items in the training set
        test_size (int): Number of items in the test set
        verbose (bool): Verbose mode
    Returns:
        -
    """
    UE.set_ue_value(UE_KEY_TRAINING_DATA_SIZE, train_size)
    UE.set_ue_value(UE_KEY_TEST_DATA_SIZE, test_size)
    ue_set_stats_mode_train(UE, verbose=verbose)
    UE.set_ue_value(UE_KEY_TRAINING_START, None)
    if verbose:
        print(f"ue_start_training: {UE.get_nametag()} train_size: {train_size},"
              f" test_size: {test_size}, stats_mode: TRAIN")

def ue_stop_training(UE, train_loss_score, verbose=False):
    """
    Stop a UE training session, store the loss score and stop the timer, storing the total
    elapsed time for the run.
    Args:
        UE (class): instance of the UEHelper class
        train_loss_score (float): training loss score
        verbose (bool): verbose mode.
    Returns:
        -
    """
    UE.set_ue_value(UE_KEY_TRAINING_END, None)
    UE.set_ue_value(UE_KEY_TRAINING_LOSS_SCORE, train_loss_score)
    if verbose:
        print(f"ue_stop_training: {UE.get_nametag()} train_loss_score: {train_loss_score}, "
              f"total training time: {UE.get_training_time()}")

def ue_start_unlearning(UE, unlearn_data_size,  verbose=False):
    """
    Start a UE unlearning session after storing the unlearn data size and starting a timer.
    Args:
        UE (class): instance of the UEHelper class
        unlearn_data_size (int): Number of items in the unlearning set
        verbose (bool): Verbose mode
    Returns:
        -
    """
    UE.set_ue_value(UE_KEY_UNLEARN_DATA_SIZE, unlearn_data_size)
    ue_set_stats_mode_unlearn(UE, verbose=verbose)
    UE.set_ue_value(UE_KEY_UNLEARN_START, None)
    if verbose:
        print(f"ue_start_unlearning: {UE.get_nametag()} unlearn_data_size: {unlearn_data_size}, "
              f"total unlearning time: {UE.get_unlearn_time()}")

def ue_stop_unlearning(UE, unlearn_loss_score, verbose=False):
    """
    Stop a UE training session, store the loss score, stop the unlearn timer and store the unlearn time.
    Args:
        UE (class): instance of the UEHelper class
        unlearn_loss_score (float): unlearning accuracy score
        verbose (bool): verbose mode.
    Returns:
        -
    """
    UE.set_ue_value(UE_KEY_UNLEARN_END, None)
    UE.set_ue_value(UE_KEY_UNLEARN_LOSS_SCORE, unlearn_loss_score)
    if verbose:
        print(f"ue_stop_unlearning: {UE.get_nametag()} unlearn_accuracy: {unlearn_loss_score}, "
              f"total unlearning time: {UE.get_unlearn_time()}")

def ue_print_run_stats(UE):
    """
    Prints the run-time stats for the last run to stdout for collecting by the wrapper.
    Args:
        UE (class): instance of the UEHelper class.
    Returns:
        -
    """
    UE.print_ue_training_values()
    UE.print_ue_unlearn_values()


def ue_save_watermark_dataset(dataset, nametag):
    """
    Saves a pytorch dataset to a named file.
    Mainly used for storing watermarked training data
    Note that a new directory is created in the model store area for each set of saved data
    Args:
        dataset (): Pytorch data set
        nametag (string): tag name for the watermark data
    Returns:
        -
    """
    datastore = f"{UE_MODEL_STORE_DIRECTORY}/{nametag}"
    if not os.path.exists(datastore):
        os.makedirs(datastore)
        for i, data in enumerate(dataset):
            torch.save(data[0], f"{datastore}/train_transformed_img{i}")
            torch.save(data[1], f"{datastore}/train_transformed_mask{i}")
    else:
        print(f"Data already stored. Please delete watermarked files in {datastore} to update")
