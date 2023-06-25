#!/usr/bin python3
# -*- coding: utf-8 -*-

import csv
from datetime import datetime
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

UE_ERROR_LOGGER = "UE_error_logger"

UE_TRAIN_MODEL = 'train'
UE_UNLEARN_MODEL = 'unlearn'
UE_TRAIN_UNLEARN = 'train_and_unlearn'
UE_VALID_MODES = [UE_TRAIN_MODEL, UE_UNLEARN_MODEL]

HOME_DIR = os.path.expanduser('~')
UE_DATA_STORE_DIRECTORY = HOME_DIR + '/unlearning_effectiveness/data'
UE_MODEL_STORE_DIRECTORY = HOME_DIR + '/unlearning_effectiveness/models'
UE_STATS_STORE_DIRECTORY = HOME_DIR + '/unlearning_effectiveness/stats'

STOREFILE_SUFFIX = "ue_store.csv"
UE_ALL_READ = "all"
VALID_WRITE_OPERATIONS = [UE_TRAIN_MODEL, UE_UNLEARN_MODEL]
VALID_READ_OPERATIONS = [UE_TRAIN_MODEL, UE_UNLEARN_MODEL, UE_ALL_READ]

UE_STATS_INTERVAL_SECS = 5
UE_GPU_STATS = 'gpu'
UE_CPU_STATS = 'cpu'

DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

class UEHelper(object):
    """
    Class to be used in third party code to interface to the Unlearning
    Effectiveness wrapper.
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

    def __init__(self, title):
        self.title = title
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

    def set_ue_value(self, keyword, value):
        if keyword == UE_KEY_CUDA:
            if value not in self.VALID_CUDA_STATES:
                self.cuda_status = 'UNKNOWN'
                return
            self.cuda_status = value
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
            if isinstance(value, str):
                try:
                    value = float(value)
                except Exception as e:
                    print(f"Loss score must be a float. Got '{value}'")
                    return
            else:
                if not isinstance(value, float):
                    print(f"Loss score must be a float. Got '{value}'")
                    return
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
            if isinstance(value, str):
                try:
                    value = float(value)
                except Exception as e:
                    print(f"Loss score must be a float. Got '{value}'")
                    return
            else:
                if not isinstance(value, float):
                    print(f"Loss score must be a float. Got '{value}'")
                    return
            self.unlearn_loss_score = value
        else:
            print(f"Unknown keyword {keyword}")
        return

    def start_training_timer(self):
        self.training_start = datetime.utcnow()#

    def start_unlearn_timer(self):
        self.unlearn_start = datetime.utcnow()

    def print_ue_training_values(self):
        """
        print current UE training values as key-value pairs to pass to the wrapper
        """
        print(f"{UE_KEY_CUDA}={self.cuda_status}")
        print(f"{UE_KEY_DATASET}={self.dataset}")
        print(f"{UE_KEY_TRAINING_DATA_SIZE}={self.training_data_size}")
        print(f"{UE_KEY_TEST_DATA_SIZE}={self.test_data_size}")
        print(f"{UE_KEY_TRAINING_TIME}={self.training_time}")
        print(f"{UE_KEY_TRAINING_LOSS_SCORE}={self.training_loss_score}")

    def print_ue_unlearn_values(self):
        """
        print current UE unlearning values as key-value pairs to pass to the wrapper
        """
        print(f"{UE_KEY_CUDA}={self.cuda_status}")
        print(f"{UE_KEY_DATASET}={self.dataset}")
        print(f"{UE_KEY_UNLEARN_DATA_SIZE}={self.unlearn_data_size}")
        print(f"{UE_KEY_UNLEARN_TIME}={self.unlearn_time}")
        print(f"{UE_KEY_UNLEARN_LOSS_SCORE}={self.unlearn_loss_score}")


def ue_print_piped_message(message):
    lines = str(message).split('/n')
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


def ue_store_model(model, filename, train_or_unlearn=UE_TRAIN_MODEL):
    if filename is not None:
        if train_or_unlearn == UE_UNLEARN_MODEL:
            filename_split = filename.split('.')
            filename = filename_split[0] + '_unlearn' + '.pt'
        torch.save(model, filename)


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


def get_stored_nametags():
    """
    Returns a list of nametags in  DATA_STORE_DIRECTORY
    Args
        -
    Returns:
        (list): List of nametags
    """
    if not os.path.exists(UE_DATA_STORE_DIRECTORY):
        os.makedirs(UE_DATA_STORE_DIRECTORY)
        return []
    filename_list = [f for f in listdir(UE_DATA_STORE_DIRECTORY) if isfile(join(UE_DATA_STORE_DIRECTORY, f))]
    nametag_list = []
    for filename in filename_list:
        nametag = filename.split('_')[0]
        nametag_list.append(nametag)
    return nametag_list, filename_list


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
    Store the score and training time for the operation.
    Args:
        nametag (string): tag name associated with this training/unlearning scenario.
        operation (string): TRAINING or UNLEARNING
        cuda_status (string): True if CUDA is enabled for the operation
        dataset (string): dataset in use.
        epochs (int): Number of epochs used in learning
        training_size (int): Number of elements in dataset used for training
        test_size (int): Number of elements in dataset used for testing
        unlearning_size (int): Number of elements in unlearning data set
        operation_score (float): Model loss score after the operation
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
    if operation not in VALID_WRITE_OPERATIONS:
        print(f"store_metrics: invalid operation {operation}, must be one of {VALID_WRITE_OPERATIONS}")
        return
    if not os.path.exists(UE_DATA_STORE_DIRECTORY):
        os.makedirs(UE_DATA_STORE_DIRECTORY)
    store_file = f"{UE_DATA_STORE_DIRECTORY}/{nametag}_{STOREFILE_SUFFIX}"
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
                     f"{gpu_peak_memory_MiB}" \
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
                           f"GPUMemPeak" \
                           f"\n"
        store_csv = store_csv_header + store_csv_data
        with open(store_file, "w") as fd:
            fd.write(store_csv)
    else:
        with open(store_file, "a") as fd:
            fd.write(store_csv_data)
    print(f'Stored metrics for {operation} tag: "{nametag}" to "{store_file}"')


def ue_display_stats(nametag, requested_operation, display_nametags):
    """
    Display the data associated with a training nametag
    Args:
        nametag (string): tag name for the model
        requested_operation (string): UE_TRAINING, UE_UNLEARNING, ALL
        display_nametags (Bool): Display nametags for available stored data
    Return:
        -
    """
    stored_nametag_list, filename_list = get_stored_nametags()
    if display_nametags:
        if len(stored_nametag_list) == 0:
            print("No test metrics are available")
            return
        print("Available nametags with stored metrics:")
        for tag in stored_nametag_list:
            print(f"    {tag}")
    if requested_operation is None:
        return
    if requested_operation not in VALID_READ_OPERATIONS:
        print(f"store_metrics: invalid operation {requested_operation}, must be one of {VALID_READ_OPERATIONS},")
        return

    if len(stored_nametag_list) == 0:
        print(f"\nNo nametag data exists in {UE_DATA_STORE_DIRECTORY}")
        return
    if nametag not in stored_nametag_list:
        print(f"\nNo metrics exist for nametag {nametag}")
    for filename in filename_list:
        output = ""
        filename_nametag = filename.split('_')[0]
        if filename_nametag == nametag:
            full_filename = f"{UE_DATA_STORE_DIRECTORY}/{filename}"
            with open(full_filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                count = 0
                for row in csv_reader:
                    if count == 0:
                        header_line = '    ' + ','.join(row)
                    stored_operation = row['operation']
                    if stored_operation == requested_operation or requested_operation == UE_ALL_READ:
                        output = output + '    ' + ','.join(row.values()) + '\n'
                    count += 1
        print(f"\nFilename: '{filename}':\n{header_line}\n{output}")


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

def ue_set_stats_type(nametag, mode=UE_TRAIN_MODEL, verbose=False):
    """
    Sets the mode for the stats updates for the nametag
    Args:
        nametag (string): name of test run
        mode (string): train or unlearn mode
        verbose (bool): verbose mode.
    """
    if mode not in UE_VALID_MODES:
        print(f"ue_set_stats_type: Bad mode {mode}")
        return
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        os.mkdir(UE_STATS_STORE_DIRECTORY)
    UE_STATS_MODE_FILE = UE_STATS_STORE_DIRECTORY + f"{nametag}-mode.dat"
    with open(UE_STATS_MODE_FILE, 'w') as mode_file:
        mode_file.write(mode)
    if verbose:
        print(f"Stats mode updated to '{mode}'")


def ue_get_stats_type(nametag, verbose=False):
    """
    Gets the current mode for the stats updates for the nametag
    Args:
        nametag (string): name of test run
        verbose (bool): verbose mode.
    Return:
        (string): current mode.
    """
    mode = UE_TRAIN_MODEL
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        ue_set_stats_type(nametag, mode=mode)
    else:
        UE_STATS_MODE_FILE = UE_STATS_STORE_DIRECTORY + f"{nametag}-mode.dat"
        with open(UE_STATS_MODE_FILE, 'r') as mode_file:
            mode = mode_file.readline()
    if mode not in UE_VALID_MODES:
        print(f"ue_get_stats_type: read bad mode {mode}")
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
        nametag (string):n ame of test run
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
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        os.makedirs(UE_STATS_STORE_DIRECTORY)
    filename = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_gpu.csv"
    with open(filename, 'w') as stats:
        stats.write(header)
    if verbose:
        print(f"Getting GPU stats using '{system_command}'")
    while True:
        process = Popen(system_command, stdout=PIPE, stderr=PIPE)
        gpu_stats, stderr = process.communicate()
        if len(stderr) != 0:
            ue_print_piped_message(f"ERROR: {stderr}")
        else:
            if not os.path.exists(UE_STATS_STORE_DIRECTORY):
                os.mkdir(UE_STATS_STORE_DIRECTORY)
            mode = ue_get_stats_type(nametag, verbose)
            filename = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_gpu.csv"
            output_line = f"{mode},{gpu_stats.decode('utf-8').replace(', ', ',').replace(' %', '').replace(' MiB', '')}"
            with open(filename, 'a') as stats:
                stats.write(output_line)
        time.sleep(interval)


def ue_get_and_store_system_stats(nametag, interval, verbose):
    """
    Get and store the current runtime stats and store in a stats file..
    This runs as a thread in parallel with a Training/unlearning operation
    gets terminated by the operation when it completes.
    Args:
        nametag (string):
        interval (int): Number of seconds between readings
        verbose (bool): verbose mode.
    """
    header = 'mode,' \
             'timestamp,' \
             'processor%,' \
             'memoryMiB\n'
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        os.makedirs(UE_STATS_STORE_DIRECTORY)
    filename = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_cpu.csv"
    with open(filename, 'w') as stats:
        stats.write(header)
    if verbose:
        print(f"Getting CPU stats")
    while True:
        timestamp = datetime.utcnow().strftime(DATETIME_FORMAT)
        cpu_pct = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory[2]
        if not os.path.exists(UE_STATS_STORE_DIRECTORY):
            os.mkdir(UE_STATS_STORE_DIRECTORY)
        mode = ue_get_stats_type(nametag, verbose)
        filename = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_cpu.csv"
        output_line = f"{mode},{timestamp},{cpu_pct},{memory_percent}\n"
        with open(filename, 'a') as stats:
            stats.write(output_line)
        time.sleep(interval)


def ue_get_gpu_cpu_stats(nametag, stats_type, operation, verbose):
    """
    Reads the stats file for the named stats type and summarises its contents to generate
    cumulative CPG/GPU, average memory and peak memory values for a run
    Args:
        nametag (string): Job nametag
        stats_type (string): CPU or GPU
        operation (string): train or unlearn operation
        verbose (bool): verbose mode.
    returns
       cumulative CPU/GPU seconds
       Average memory usage (MB)
       Peak memory usage (MB)
    """
    if operation not in [VALID_READ_OPERATIONS]:
        print(f"ue_get_gpu_cpu_stats: invalid operation '{operation}'")
        return
    if not os.path.exists(UE_STATS_STORE_DIRECTORY):
        os.makedirs(UE_STATS_STORE_DIRECTORY)
    if stats_type == UE_CPU_STATS:
        filename = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_cpu.csv"
    elif stats_type == UE_GPU_STATS:
        filename = f"{UE_STATS_STORE_DIRECTORY}/{nametag}_gpu.csv"
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
    # The processor % is the % of CPU/GPU time used in the last STATS_INTERVAL seconds.
    # The number of seconds of CPU time in this interval is calculated as below
    cumulative_processor_seconds = (int(df_stats['processor%'].sum()) * UE_STATS_INTERVAL_SECS) / 100
    avg_memory_MiB = round(int(df_stats['memoryMiB'].sum()) / count, 2)
    peak_memory_MiB = int(df_stats['memoryMiB'].max())
    if verbose:
        print(f"get_stats: {nametag}: ")
    return cumulative_processor_seconds, avg_memory_MiB, peak_memory_MiB
