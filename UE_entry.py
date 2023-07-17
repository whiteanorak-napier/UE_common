#!/usr/bin python3
# -*- coding: utf-8 -*-

"""
Unlearning Effectiveness (UE) module used to execute 3rd party code. Can be used to perform train, unlearn
and membership inference operations on the target code. Gathers system stats (elapsed time, CPU time, average
CPU memory usage, GPU time, GPU memory usage) during execution. When the call is complete, stores the resultant
stats along with the results including the training loss score, the loss score after unlearning and the membership
inference score
"""

import argparse
from datetime import timedelta
import sys
import os
import multiprocessing
from subprocess import Popen, PIPE
from torchvision.datasets import MNIST, CIFAR10

from UE_interface import (
    ue_extract_token_value,
    UE_KEY_CUDA,
    UE_KEY_DATASET,
    UE_KEY_TRAINING_DATA_SIZE,
    UE_KEY_TEST_DATA_SIZE,
    UE_KEY_TRAINING_TIME,
    UE_KEY_TRAINING_LOSS_SCORE,
    UE_KEY_UNLEARN_DATA_SIZE,
    UE_KEY_UNLEARN_TIME,
    UE_KEY_UNLEARN_LOSS_SCORE,
    UE_KEY_INFERENCE,

    UE_MODEL_STORE_DIRECTORY,
    UE_CPU_STATS,
    UE_GPU_STATS,
    UE_STATS_INTERVAL_SECS,

    UE_OPERATION_TRAIN,
    UE_OPERATION_UNLEARN,
    UE_OPERATION_INFERENCE,
    UE_OPERATION_TRAIN_UNLEARN,
    UE_OPERATION_WATERMARK,
    UE_OPERATION_DISPLAY_TAGS,
    UE_OPERATION_DISPLAY_STATS,
    UE_VALID_OPERATIONS,

    ue_get_stored_nametags,
    ue_get_effectiveness_stats,
    ue_get_files_in_directory,
    ue_get_gpu_cpu_stats,
    ue_set_stats_mode_unlearn,
    ue_set_stats_mode_train,
    ue_get_and_store_gpu_stats,
    ue_get_and_store_system_stats,
    ue_print_formatted_bytestring,
    ue_store_metrics,
    ue_display_stats_and_generate_mue
)

DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
SHARED_DIR = "/tmp"
LOG_DIR = f"{SHARED_DIR}/log_files"
DATASETS = [MNIST, CIFAR10]

BASE_DIRECTORY = os.getcwd()
EXECUTABLE_PATH = f"{BASE_DIRECTORY}/"

EXECUTION_TRAIN = 'train'
EXECUTION_UNLEARN = 'unlearn'
EXECUTION_TRAIN_UNLEARN = 'train_unlearn'
EXECUTION_INFERENCE = 'inference'
EXECUTION_WATERMARK = 'watermark'
RUN_SCRIPT = 'executable'
RUN_ARGS = 'args'

"""
############################################################
Start of section to be changed for each 3rd party code base
TRAIN_SCRIPT          - name of training script
UNLEARN_SCRIPT        - name of unlearning script
TRAIN_UNLEARN_SCRIPT  - name of script to train and unlearn
                        in a single operation
INFERENCE_SCRIPT      - name of script used for membership
                        inference tests
WATERMARK_SCRIPT      - name of script used to train a 
                        watermarked data set.
#############################################################
"""
TRAIN_SCRIPT = f""
UNLEARN_SCRIPT = f""
TRAIN_UNLEARN_SCRIPT = f""
INFERENCE_SCRIPT = f""
WATERMARK_SCRIPT = f"UE_train_watermarked.py"

EXECUTION_COMMANDS = {
    EXECUTION_TRAIN: {
        RUN_SCRIPT: TRAIN_SCRIPT,
        RUN_ARGS: []
    },
    EXECUTION_UNLEARN: {
        RUN_SCRIPT: UNLEARN_SCRIPT,
        RUN_ARGS: []
    },
    EXECUTION_TRAIN_UNLEARN: {
        RUN_SCRIPT: TRAIN_UNLEARN_SCRIPT,
        RUN_ARGS: []
    },
    EXECUTION_INFERENCE: {
        RUN_SCRIPT: INFERENCE_SCRIPT,
        RUN_ARGS: []
    },
    EXECUTION_WATERMARK: {
        RUN_SCRIPT: WATERMARK_SCRIPT,
        RUN_ARGS: ["--gpu-id", "0",
                   "--checkpoint", "checkpoint/watermark"]
    }
}
##############################################################
# End of section to be configured for each 3rd party code base
# Code beyond this point may also be modified to fit with the
# arguments required by the target code.
##############################################################


def train_model(nametag, epochs, gpu_collector, cpu_collector, verbose):
    """
    Calls 3rd party code.
    Creates a processes to train a model and when complete, terminates the parallel CPU and GPU collector
    processes and stores the resulting stats in UE storage.
    Can request that the resultant model is stored in UE storage if the 3rd party code has
    the --UE_store_model_filename option configured.
    Args:
        nametag (string): tag name for the model
        epochs (int): Number of epochs to run
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        removal_mode (string): one of "feature", "node" or "edge" (3rd part code specific parameter)
        verbose (bool): Verbose trace
    :returns:
        -
    """

    # Note that Popen requires each argument to be passed as a separate string in a list.
    if not os.path.exists(UE_MODEL_STORE_DIRECTORY):
        os.makedirs(UE_MODEL_STORE_DIRECTORY)
    existing_models = ue_get_files_in_directory(UE_MODEL_STORE_DIRECTORY)
    requested_model = f"{nametag}.pt"
    if requested_model not in existing_models:
        print(f"Model {requested_model} will be created in {UE_MODEL_STORE_DIRECTORY}")
    else:
        print(f"Model {requested_model} already exists in {UE_MODEL_STORE_DIRECTORY} and will be overwritten")
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_TRAIN][RUN_SCRIPT]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_TRAIN][RUN_ARGS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params +  \
        ["--UE_store_model_filename", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}",
         "--epochs", f"{epochs}"]

    if verbose:
        print(f"Calling 3rd party training executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)

    # Block and wait for the process to complete
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_formatted_bytestring("Error Message:", stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            ue_print_formatted_bytestring("Command output:", stderr)
            # Extract stats from stdout.
        cuda_status = ue_extract_token_value(string_stdout, UE_KEY_CUDA)
        dataset = ue_extract_token_value(string_stdout, UE_KEY_DATASET)
        training_size = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_DATA_SIZE)
        test_size = ue_extract_token_value(string_stdout, UE_KEY_TEST_DATA_SIZE)
        training_time = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_TIME)
        training_score = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_LOSS_SCORE)
        if verbose:
            print(f"Training Time = {training_time}, Training Score = {training_score}")
    except Exception as e:
        print(f"Unable to gather stats from code execution, error '{e}'")
        exit(1)
    num_removes = 0
    cpu_cumulative_seconds, cpu_average_memory_MiB, cpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_OPERATION_TRAIN, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_OPERATION_TRAIN, verbose)
    ue_store_metrics(nametag,
                     UE_OPERATION_TRAIN,
                     cuda_status,
                     dataset,
                     epochs,
                     training_size,
                     test_size,
                     num_removes,
                     training_score,
                     training_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB,
                     )


def unlearn_model(nametag, num_removes, gpu_collector, cpu_collector, verbose):
    """
    Calls 3rd party code.
    Creates a processes to unlearn data from a model and when complete, terminates the
    parallel CPU and GPU collector processes and stores the resulting stats in UE storage.
    Can request that the resultant model is stored in UE storage if the 3rd party code has
    the --UE_store_model_filename option configured.
    Args:
        nametag (string): tag name for the model
        num_removes (int): Number of items to remove
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        removal_mode (string): one of "feature", "node" or "edge" (3rd part code specific parameter)
        verbose (bool): Verbose trace
    :returns:
        -
    """

    # Note that Popen requires each argument to be passed as a separate string in a list.
    if not os.path.exists(UE_MODEL_STORE_DIRECTORY):
        os.makedirs(UE_MODEL_STORE_DIRECTORY)
    existing_models = ue_get_files_in_directory(UE_MODEL_STORE_DIRECTORY)
    requested_model = f"{nametag}.pt"
    if requested_model not in existing_models:
        print(f"Model {requested_model} will be created in {UE_MODEL_STORE_DIRECTORY}")
    else:
        print(f"Model {requested_model} already exists in {UE_MODEL_STORE_DIRECTORY} and will be overwritten")
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][RUN_SCRIPT]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][RUN_ARGS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params + \
        ["--UE_store_model_filename", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}"]

    if verbose:
        print(f"Calling 3rd party training executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)

    # Block and wait for the process to complete
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_formatted_bytestring("Error Message:", stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            ue_print_formatted_bytestring("Command output:", stderr)
            # Extract stats from stdout.
        cuda_status = ue_extract_token_value(string_stdout, UE_KEY_CUDA)
        dataset = ue_extract_token_value(string_stdout, UE_KEY_DATASET)
        unlearning_size = ue_extract_token_value(string_stdout, UE_KEY_UNLEARN_DATA_SIZE)
        unlearning_time = ue_extract_token_value(string_stdout, UE_KEY_UNLEARN_TIME)
        unlearning_score = ue_extract_token_value(string_stdout, UE_KEY_UNLEARN_LOSS_SCORE)
        if verbose:
            print(f"Unlearning Time = {unlearning_time}, Unlearning Score = {unlearning_score}")

    except Exception as e:
        print(f"Unable to gather stats from code execution, error '{e}'")
        exit(1)

    cpu_cumulative_seconds, cpu_average_memory_MiB, cpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_OPERATION_UNLEARN, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_OPERATION_UNLEARN, verbose)
    test_size = 0
    training_size = 0

    ue_store_metrics(nametag,
                     UE_OPERATION_UNLEARN,
                     cuda_status,
                     dataset,
                     num_removes,
                     training_size,
                     test_size,
                     unlearning_size,
                     unlearning_score,
                     unlearning_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB,
                     )


def train_and_unlearn(nametag, num_removes, removal_mode, gpu_collector, cpu_collector, verbose):
    """
    *** Codebase specific. ***
    **************************
    For use with 3rd party code where the train and unlearn activities are performed in the same code segment.
    Creates a processes to train/unlearn, and when complete, terminates the parallel CPU and GPU collector
    processes and stores the resulting stats in UE storage.
    Can request that the resultant model is stored in UE storage if the 3rd party code has
    the --UE_store_model_filename option configured.
    Args:
        nametag (string): tag name for the model
        num_removes (int): Number of elements to remove in unlearning
        removal_mode (string): one of "feature", "node" or "edge" (codebase specific, not used here.)
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        verbose (bool): Verbose trace
    :returns:
        -
    """

    # Note that Popen requires each argument to be passed as a separate string in a list.
    if not os.path.exists(UE_MODEL_STORE_DIRECTORY):
        os.makedirs(UE_MODEL_STORE_DIRECTORY)
    existing_models = ue_get_files_in_directory(UE_MODEL_STORE_DIRECTORY)
    requested_model = f"{nametag}.pt"
    if requested_model not in existing_models:
        print(f"Model {requested_model} will be created in {UE_MODEL_STORE_DIRECTORY}")
    else:
        print(f"Model {requested_model} already exists in {UE_MODEL_STORE_DIRECTORY} and will be overwritten")

    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_TRAIN_UNLEARN][RUN_SCRIPT]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_TRAIN_UNLEARN][RUN_ARGS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params + \
        ["--UE_store_model_filename", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}",
         "--num_clusters", f"{num_removes}"]

    if verbose:
        print(f"Calling 3rd party executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)

    # Block and wait for the process to complete
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_formatted_bytestring("Error Message:", stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            ue_print_formatted_bytestring("Command output:", stderr)
            # Extract stats from stdout.
        cuda_status = ue_extract_token_value(string_stdout, UE_KEY_CUDA)
        dataset = ue_extract_token_value(string_stdout, UE_KEY_DATASET)
        training_size = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_DATA_SIZE)
        test_size = ue_extract_token_value(string_stdout, UE_KEY_TEST_DATA_SIZE)
        training_time = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_TIME)
        training_score = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_LOSS_SCORE)
        unlearning_size = ue_extract_token_value(string_stdout, UE_KEY_UNLEARN_DATA_SIZE)
        unlearning_time = ue_extract_token_value(string_stdout, UE_KEY_UNLEARN_TIME)
        unlearning_score = ue_extract_token_value(string_stdout, UE_KEY_UNLEARN_LOSS_SCORE)
        if verbose:
            print(f"Training Time = {training_time}, Training Score = {training_score}")
    except Exception as e:
        print(f"Unable to gather stats from code execution, error {e}")
        exit(1)

    cpu_cumulative_seconds, cpu_average_memory_MiB, cpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_OPERATION_TRAIN, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_OPERATION_TRAIN, verbose)
    ue_store_metrics(nametag,
                     UE_OPERATION_TRAIN,
                     cuda_status,
                     dataset,
                     num_removes,
                     training_size,
                     test_size,
                     unlearning_size,
                     training_score,
                     training_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB,
                     )
    cpu_cumulative_seconds, cpu_average_memory_MiB, cpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_OPERATION_UNLEARN, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_OPERATION_UNLEARN, verbose)
    ue_store_metrics(nametag,
                     UE_OPERATION_UNLEARN,
                     cuda_status,
                     dataset,
                     num_removes,
                     training_size,
                     test_size,
                     unlearning_size,
                     unlearning_score,
                     unlearning_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB,
                     )


def membership_inference(nametag, unlearned_data, verbose):
    """
    For use with 3rd party code to infer if the unlearned_data exists in a model.
    Args:
        nametag (string): tag name for the model
        unlearned_data (int): Data to test for in the model
        verbose (bool): Verbose mode
    :returns:
        cuda_status (string) True if CUDA is enabled in the target code.
        dataset (string): Name of dataset being used
        inference_score (float): Percentage of unlearned data detected in the mode.
    """
    # Note that Popen requires each argument to be passed as a separate string in a list.
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][RUN_SCRIPT]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][RUN_ARGS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params + \
        ["--UE_nametag", f"{nametag}"]

    if verbose:
        print(f"Calling 3rd party membership inference executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if len(stderr) != 0:
        ue_print_formatted_bytestring("Error Message:", stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            ue_print_formatted_bytestring("Command output:", stderr)
            # Extract stats from stdout.
        inference_score = ue_extract_token_value(string_stdout, UE_KEY_INFERENCE)
        if verbose:
            print(f"inference_score= {inference_score}")

    except Exception as e:
        print(f"Unable to gather stats from code execution, error '{e}'")
        exit(1)

    cuda_status = "-"
    dataset = "-"
    test_size = 0
    training_size = 0
    unlearning_size = 0
    unlearning_time = timedelta(days=0)
    cpu_cumulative_seconds = 0.0
    cpu_average_memory_MiB = 0.0
    cpu_peak_memory_MiB = 0.0
    gpu_cumulative_seconds = 0.0
    gpu_average_memory_MiB = 0.0
    gpu_peak_memory_MiB = 0.0
    num_removes = 0
    ue_store_metrics(nametag,
                     UE_OPERATION_INFERENCE,
                     cuda_status,
                     dataset,
                     num_removes,
                     training_size,
                     test_size,
                     unlearning_size,
                     inference_score,
                     unlearning_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB,
                     )

def train_watermarked_model(nametag, epochs, gpu_collector, cpu_collector, verbose):
    """
    Calls 3rd party code.
    Creates a processes to train a watermarked model and when complete, terminates the
    parallel CPU and GPU collector processes and stores the resulting stats in UE storage.
    Requests that the resultant model and watermarked data is stored in UE storage.
    Args:
        nametag (string): tag name for the model
        epochs (int): Number of epochs to run
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        verbose (bool): Verbose trace
    :returns:
        -
    """
    # Note that Popen requires each argument to be passed as a separate string in a list.
    if not os.path.exists(UE_MODEL_STORE_DIRECTORY):
        os.makedirs(UE_MODEL_STORE_DIRECTORY)
    existing_models = ue_get_files_in_directory(UE_MODEL_STORE_DIRECTORY)
    requested_model = f"{nametag}.pt"
    watermarked_data = f"{nametag}_watermarked.data"
    if requested_model not in existing_models:
        print(f"Model {requested_model} will be created in {UE_MODEL_STORE_DIRECTORY}")
    else:
        print(f"Model {requested_model} already exists in {UE_MODEL_STORE_DIRECTORY} and will be overwritten")
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_WATERMARK][RUN_SCRIPT]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_WATERMARK][RUN_ARGS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params +  \
        ["--UE_store_model_filename", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_store_watermarked_data", f"{watermarked_data}",
         "--UE_nametag", f"{nametag}",
         f"--epochs", f"{epochs}"
         ]

    if verbose:
        print(f"Calling 3rd party training executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)

    # Block and wait for the process to complete
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_formatted_bytestring("Error message:", stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            ue_print_formatted_bytestring("Command output:", stderr)
            # Extract stats from stdout.
        cuda_status = ue_extract_token_value(string_stdout, UE_KEY_CUDA)
        dataset = ue_extract_token_value(string_stdout, UE_KEY_DATASET)
        training_size = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_DATA_SIZE)
        test_size = ue_extract_token_value(string_stdout, UE_KEY_TEST_DATA_SIZE)
        training_time = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_TIME)
        training_score = ue_extract_token_value(string_stdout, UE_KEY_TRAINING_LOSS_SCORE)
        if verbose:
            print(f"Training Time = {training_time}, Training Score = {training_score}")
    except Exception as e:
        print(f"Unable to gather stats from code execution, error '{e}'")
        exit(1)
    num_removes = 0
    cpu_cumulative_seconds, cpu_average_memory_MiB, cpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_OPERATION_TRAIN, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_OPERATION_TRAIN, verbose)
    ue_store_metrics(nametag,
                     UE_OPERATION_TRAIN,
                     cuda_status,
                     dataset,
                     epochs,
                     training_size,
                     test_size,
                     num_removes,
                     training_score,
                     training_time,
                     cpu_cumulative_seconds,
                     cpu_average_memory_MiB,
                     cpu_peak_memory_MiB,
                     gpu_cumulative_seconds,
                     gpu_average_memory_MiB,
                     gpu_peak_memory_MiB,
                     )



def process_and_gather_stats(operation, nametag, epochs, num_removes, removal_mode, verbose):
    """
    Perform the operation (training, unlearing, membership inference or watermark generation)
    using the configured external script as an entry point, storing temporary and results
    data in files named for the nametag. "Epochs" is used for learning, "num_removes" for unlearning.
    The operation is kicked off in parallel with two additional processes, one to gather GPU and
    GPU memory stats, and one to gather CPU and mCPU emory stats.
    Args:
         operation (string): One of train, unlearn or membership inference
         nametag (string): tag name for the model
         epochs (int): Number of training epochs
         num_removes (int): Number of elements to remove in unlearning
         removal_mode (string): one of "feature", "node" or "edge" (as examples)
         verbose (bool): Verbose trace
    Returns:
        -
    """
    if operation in [UE_OPERATION_UNLEARN]:
        ue_set_stats_mode_unlearn(nametag)
    else:
        ue_set_stats_mode_train(nametag)
    print("Starting process to gather GPU stats")
    gpu_collector = \
        multiprocessing.Process(target=ue_get_and_store_gpu_stats, args=(nametag, UE_STATS_INTERVAL_SECS, False))
    gpu_collector.start()
    print("Starting process to gather CPU stats")
    cpu_collector = \
        multiprocessing.Process(target=ue_get_and_store_system_stats, args=(nametag, UE_STATS_INTERVAL_SECS, False))
    cpu_collector.start()
    if operation == UE_OPERATION_TRAIN:
        print(f"Starting process to train the model")
        run_task = multiprocessing.Process(target=train_model,
                                           args=(nametag,
                                                 epochs,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    elif operation == UE_OPERATION_UNLEARN:
        print("Starting process to unlearn the model")
        run_task = multiprocessing.Process(target=unlearn_model,
                                           args=(nametag,
                                                 num_removes,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    elif operation == UE_OPERATION_INFERENCE:
        print("Starting process to perform membership inference on the model")
        run_task = multiprocessing.Process(target=membership_inference,
                                           args=(nametag,
                                                 num_removes,
                                                 removal_mode,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    elif operation == UE_OPERATION_WATERMARK:
        print("Starting process to perform membership inference on the model")
        run_task = multiprocessing.Process(target=train_watermarked_model,
                                           args=(nametag,
                                                 epochs,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    else:
        print(f"Unknown operation {operation}. Exiting")
        sys.exit(1)
    run_task.start()
    run_task.join()
    print(f"Operation '{operation}' has completed")


def main():
    (
        nametag,
        operation,
        num_removes,
        removal_mode,
        epochs,
        verbose,
    ) = check_args((sys.argv[1:]))

    msg = (
        f"nametag:                  {nametag}\n"
        f"operation:                {operation}\n"
        f"removal_mode:             {removal_mode}\n"
        f"num_removes:              {num_removes}\n"
        f"epochs:                   {epochs}\n"
        f"verbose:                  {verbose}\n"
    )
    print(msg)

    if operation not in UE_VALID_OPERATIONS:
        print(f"Invalid operation {operation}")
        sys.exit(1)

    # Display operations
    if operation == UE_OPERATION_DISPLAY_TAGS:
        ue_get_stored_nametags(True)
        sys.exit(0)

    if nametag is None:
        print(f"A nametag must be supplied for a {operation} operation")
        sys.exit(1)

    if operation == UE_OPERATION_DISPLAY_STATS:
        stats = ue_get_effectiveness_stats(nametag)
        ue_display_stats_and_generate_mue(nametag, stats)
        sys.exit(0)
    # Processing operations
    process_and_gather_stats(operation, nametag, epochs, num_removes, removal_mode, verbose)


def check_args(args=None):

    parser = argparse.ArgumentParser(description='Unlearning Effectiveness entry point')

    parser.add_argument(
        '-tag', '--nametag',
        help='Name or tag of model to train or unlearn from.',
        required=False,
        default=None
    )

    parser.add_argument(
        '-o', '--operation',
        help='Operation to perform. Must be one of train, unlearn, inference, display_tags or display_stats',
        required=False,
        default=UE_OPERATION_TRAIN,
    )

    parser.add_argument(
        '-nr', '--num_removes',
        help='Set number of elements to remove during unlearning. Default 800',
        required=False,
        default=800
    )

    parser.add_argument(
        '-rm', '--removal_mode',
        help='Codebase specific - Set removal mode - examples are "feature", "node" or "edge". Default is "feature"',
        required=False,
        default='feature'
    )

    parser.add_argument(
        '-e', '--epochs',
        help='Codebase specific - Set number of items for removal, default 150',
        required=False,
        default=150
    )

    parser.add_argument(
        '-v', '--verbose',
        help='Display runtime trace',
        required=False,
        default=None,
        action='store_true'
    )

    cmd_line_args = parser.parse_args(args)

    return(
        cmd_line_args.nametag,
        cmd_line_args.operation,
        cmd_line_args.num_removes,
        cmd_line_args.removal_mode,
        cmd_line_args.epochs,
        cmd_line_args.verbose,
    )


if __name__ == '__main__':
    main()
