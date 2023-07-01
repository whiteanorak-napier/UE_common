#!/usr/bin python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import multiprocessing
from subprocess import Popen, PIPE
import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10

from UE_helper import (
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

    UE_MODEL_STORE_DIRECTORY,
    UE_CPU_STATS,
    UE_GPU_STATS,
    UE_STATS_INTERVAL_SECS,
    UE_TRAIN_MODEL,
    UE_UNLEARN_MODEL,
    UE_OPERATION_TRAIN_UNLEARN,

    UE_OPERATION_TRAIN,
    UE_OPERATION_UNLEARN,
    UE_OPERATION_INFERENCE,
    UE_VALID_OPERATIONS,

    ue_display_stats,
    ue_get_files_in_directory,
    ue_get_gpu_cpu_stats,
    ue_get_and_store_gpu_stats,
    ue_get_and_store_system_stats,
    ue_print_piped_message,
    ue_store_metrics,
)



DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
SHARED_DIR = "/tmp"
LOG_DIR = f"{SHARED_DIR}/log_files"
DATASETS = [MNIST, CIFAR10]

BASE_DIRECTORY = os.getcwd()
EXECUTABLE_PATH = f"{BASE_DIRECTORY}/"
TRAIN_EXECUTABLE = f""
UNLEARN_EXECUTABLE = f""
TRAIN_UNLEARN_EXECUTABLE = f""
INFERENCE_EXECUTABLE = f""
EXECUTION_TRAIN = 'train'
EXECUTION_UNLEARN = 'unlearn'
EXECUTION_TRAIN_UNLEARN = 'train_unlearn'
EXECUTION_INFERENCE = 'inference'

EXECUTION_COMMANDS = {
    EXECUTION_TRAIN: {
        'executable': TRAIN_EXECUTABLE,
        'commands': []
    },
    EXECUTION_UNLEARN: {
        'executable': UNLEARN_EXECUTABLE,
        'commands': []
    },
    EXECUTION_TRAIN_UNLEARN: {
        'executable': TRAIN_UNLEARN_EXECUTABLE,
        'commands': []
    },
    EXECUTION_INFERENCE: {
        'executable': INFERENCE_EXECUTABLE,
        'commands': []
    }
}


def display_errors(errors):
    for line in errors.split("\n"):
        print(f"ERROR: {line}")

def get_training_data(dataset, data_start, data_count, verbose):
    """
    Returns an iterator for the specified dataset starting at an offset of data_start into the data
    and running up to data_count.
    Args:
        dataset (string): name of dataset
        data_start (int): Start index into dataset
        data_count (int): Number of data items in iterator
        verbose (bool): verbose mode
    Returns:
        (iterator): Training data or None if an unknown dataset is passed.
    """
    if dataset == MNIST.__name__:
        train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=None)
        if data_count is None:
            filtered_train_set = train_set
        else:
            data_filter = list(range(data_start, data_start + data_count))
            filtered_train_set = torch.utils.data.Subset(train_set, data_filter)
        return filtered_train_set
    print(f"{dataset} is not available")
    return None

def train_model(nametag, epochs, gpu_collector, cpu_collector, verbose):
    """
    For use with 3rd party code to train a model
    Args:
        nametag (string): tag name for the model
        epochs (int): Number of epochs to run
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        removal_mode (string): one of "feature", "node" or "edge" (3rd part code specific parameter)
        verbose (bool): Verbose trace
    :returns:
        cuda_status (string) True if CUDA is enabled in the target code.
        dataset (string): Name of dataset being used
        training_size (int): size of training dataset
        test_size (int): size of test dataset
        training_time (string): Time taken to train
        training_score (float): Loss score after training.
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
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_TRAIN][EXECUTION_UNLEARN]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_TRAIN][EXECUTION_COMMANDS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params +  \
        ["--UE_store_model", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}",
         "--epochs", f"{epochs}"]

    if verbose:
        print(f"Calling 3rd party training executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_piped_message(stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            print(stdout)
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
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_TRAIN_MODEL, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_TRAIN_MODEL, verbose)
    ue_store_metrics(nametag,
                     UE_TRAIN_MODEL,
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
                     gpu_peak_memory_MiB
                     )


def unlearn_model(nametag, num_removes, gpu_collector, cpu_collector, verbose):
    """
    For use with 3rd party code to unlearn a model
    Args:
        nametag (string): tag name for the model
        num_removes (int): Number of items to remove
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        removal_mode (string): one of "feature", "node" or "edge" (3rd part code specific parameter)
        verbose (bool): Verbose trace
    :returns:
        cuda_status (string) True if CUDA is enabled in the target code.
        dataset (string): Name of dataset being used
        training_size (int): size of training dataset
        test_size (int): size of test dataset
        training_time (string): Time taken to train
        training_score (float): Loss score after training.
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
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][EXECUTION_UNLEARN]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][EXECUTION_COMMANDS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params + \
        ["--UE_store_model", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}",
         "--num_removes", f"{num_removes}"]

    if verbose:
        print(f"Calling 3rd party training executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_piped_message(stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            print(stdout)
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
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_UNLEARN_MODEL, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_UNLEARN_MODEL, verbose)
    test_size = 0
    training_size = 0

    ue_store_metrics(nametag,
                     UE_UNLEARN_MODEL,
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
                     gpu_peak_memory_MiB
                     )


def train_and_unlearn(nametag, num_removes, removal_mode, gpu_collector, cpu_collector, verbose):
    """
    For use with 3rd party code where the train and unlearn activities are performed in the same code segment
    Wrapper to train and unlearn
    Per-codebase specific.
    Args:
        nametag (string): tag name for the model
        executable_filename (string): Name of file to run to perform the training.
        num_removes (int): Number of elements to remove in unlearning
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        removal_mode (string): one of "feature", "node" or "edge"
        verbose (bool): Verbose trace
    :returns:
        cuda_status (string) True if CUDA is enabled in the target code.
        dataset (string): Name of dataset being used
        training_size (int): size of training dataset
        test_size (int): size of test dataset
        training_time (string): Time taken to train
        training_score (float): Loss score after training.
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

    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_TRAIN_UNLEARN]['executable']
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_TRAIN_UNLEARN]['commands']
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params + \
        ["--UE_store_model", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}",
         "--num_removes", f"{num_removes}"]

    if verbose:
        print(f"Calling 3rd party executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_piped_message(stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            print(stdout)
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
        print("Unable to gather stats from code execution")
        exit(1)

    cpu_cumulative_seconds, cpu_average_memory_MiB, cpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_TRAIN_MODEL, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_TRAIN_MODEL, verbose)
    ue_store_metrics(nametag,
                     UE_UNLEARN_MODEL,
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
                     gpu_peak_memory_MiB
                     )


def inference(nametag, num_removes, gpu_collector, cpu_collector, verbose):
    """
    For use with 3rd party code to infer if data exists in a model.
    TODO
    Args:
        nametag (string): tag name for the model
        num_removes (int): Number of items to remove
        gpu_collector (thread): To be terminated at end of run
        cpu_collector (thread): To be terminated at end of run
        removal_mode (string): one of "feature", "node" or "edge" (3rd part code specific parameter)
        verbose (bool): Verbose trace
    :returns:
        cuda_status (string) True if CUDA is enabled in the target code.
        dataset (string): Name of dataset being used
        training_size (int): size of training dataset
        test_size (int): size of test dataset
        training_time (string): Time taken to train
        training_score (float): Loss score after training.
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
    executable_filename = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][EXECUTION_UNLEARN]
    executable_params = EXECUTION_COMMANDS[UE_OPERATION_UNLEARN][EXECUTION_COMMANDS]
    system_command = \
        ["python3",
         f"{executable_filename}"] + \
        executable_params + \
        ["--UE_store_model", f"{UE_MODEL_STORE_DIRECTORY}/{requested_model}",
         "--UE_nametag", f"{nametag}",
         "--num_removes", f"{num_removes}"]

    if verbose:
        print(f"Calling 3rd party training executable using '{system_command}'")
    process = Popen(system_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print("Stopping GPU & CPU collectors")
    gpu_collector.terminate()
    cpu_collector.terminate()
    if len(stderr) != 0:
        ue_print_piped_message(stderr)
    try:
        string_stdout = str(stdout)
        if verbose:
            print(stdout)
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
        ue_get_gpu_cpu_stats(nametag, UE_CPU_STATS, UE_UNLEARN_MODEL, verbose)
    gpu_cumulative_seconds, gpu_average_memory_MiB, gpu_peak_memory_MiB = \
        ue_get_gpu_cpu_stats(nametag, UE_GPU_STATS, UE_UNLEARN_MODEL, verbose)
    test_size = 0
    training_size = 0

    ue_store_metrics(nametag,
                     UE_UNLEARN_MODEL,
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
                     gpu_peak_memory_MiB
                     )



def process_and_gather_stats(operation, nametag, epochs, num_removes, removal_mode, verbose):
    """
    Perform the operation (TRAIN or UNLEARN) using the executable_filename, storing data
    in files named for the nametag. Epochs is used for learning.
    The operation is kicked off in parallel with two additional processes, one to gather GPU stats
    and one to gather stats from the rest of the system.
    Args:
         operation (string): One of TRAIN or UNLEARN
         nametag (string): tag name for the model
         epochs (int): Number of training epochs
         num_removes (int): Number of elements to remove in unlearning
         removal_mode (string): one of "feature", "node" or "edge"
         verbose (bool): Verbose trace
    Returns:
        cuda_status (string) True if CUDA is enabled in the target code.
        dataset (string): Name of dataset being used
        training_size (int): size of training dataset
        test_size (int): size of test dataset
        training_time (string): Time taken to train
        training_score (float): Loss score after training.
    """

    print("Starting process to gather GPU stats")
    gpu_collector = \
        multiprocessing.Process(target=ue_get_and_store_gpu_stats, args=(nametag, UE_STATS_INTERVAL_SECS, False))
    gpu_collector.start()
    print("Starting process to gather CPU stats")
    cpu_collector = \
        multiprocessing.Process(target=ue_get_and_store_system_stats, args=(nametag, UE_STATS_INTERVAL_SECS, False))
    cpu_collector.start()
    if operation == UE_OPERATION_TRAIN:
        print("Starting process to train the model")
        run_task = multiprocessing.Process(target=train_model,
                                           args=(nametag,
                                                 epochs,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    elif operation == UE_OPERATION_UNLEARN:
        print("Starting process to train the model")
        run_task = multiprocessing.Process(target=unlearn_model,
                                           args=(nametag,
                                                 num_removes,
                                                 removal_mode,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    elif operation == UE_OPERATION_INFERENCE:
        print("Starting process to train the model")
        run_task = multiprocessing.Process(target=inference,
                                           args=(nametag,
                                                 num_removes,
                                                 removal_mode,
                                                 gpu_collector,
                                                 cpu_collector,
                                                 verbose))
    else:
        print(f"Unknown operation {operation}. Exiting")
        sys.exit(1)
    run_task.start()
    run_task.join()
    print(f"'{operation} has completed")

def main():
    (
        list_datasets,
        operation,
        dataset,
        nametag,
        unlearn,
        display_metrics,
        display_metrics_nametags,
        display_inference_score,
        epochs,
        num_removes,
        removal_mode,
        verbose,
    ) = check_args((sys.argv[1:]))

    msg = (
        f"list_datasets:            {list_datasets}\n"
        f"operation:                {operation}\n"
        f"dataset:                  {dataset}\n"
        f"nametag:                  {nametag}\n"
        f"unlearn:                  {unlearn}\n"
        f"display_metrics:          {display_metrics}\n"
        f"display_metrics_nametags: {display_metrics_nametags}\n"
        f"display_inference_score:  {display_inference_score}\n"
        f"epochs:                   {epochs}\n"
        f"num_removes:              {num_removes}\n"
        f"removal_mode:             {removal_mode}\n"
        f"verbose:                  {verbose}\n"
    )
    print(msg)

    if display_metrics is not None or display_metrics_nametags:
        ue_display_stats(nametag, display_metrics, display_metrics_nametags)
        sys.exit(0)

    if operation not in UE_VALID_OPERATIONS:
        print(f"Invalid operation {operation}")
        sys.exit(1)

    process_and_gather_stats(operation, nametag, epochs, num_removes, removal_mode, verbose)


def check_args(args=None):

    parser = argparse.ArgumentParser(description='docker wrapper')

    parser.add_argument(
        '-ls', '--list_datasets',
        help='List the names of datasets that this container can use',
        required=False,
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '-o', '--operation',
        help='Operation to perform. Must be one of train or unlearn.',
        required=False,
        default=UE_OPERATION_TRAIN,
    )

    parser.add_argument(
        '-data', '--dataset',
        help='Name of dataset to use for training. Default is CIFAR10 for this wrapper',
        required=False,
        default=CIFAR10
    )

    parser.add_argument(
        '-tag', '--nametag',
        help='Name or tag of model to train or unlearn from.',
        required=True,
        default=None
    )

    parser.add_argument(
        '-ul', '--unlearn',
        help='Unlearn the data in the specified filename from the nametag model',
        required=False,
        default=None
    )

    parser.add_argument(
        '-dm', '--display_metrics',
        help='Display metrics - can use training, unlearning or all',
        required=False,
        default=None
    )

    parser.add_argument(
        '-dn', '--display_metrics_nametags',
        help='Display the nametags for stored metrics',
        required=False,
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '-dis', '--display_inference_score',
        help='Display the membership inference score for an unlearnt model',
        required=False,
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '-nr', '--num_removes',
        help='Set number of elements to remove during unlearning. Default 800',
        required=False,
        default=800
    )

    parser.add_argument(
        '-rm', '--removal_mode',
        help='Set removal mode - can be "feature", "node" or "edge". Default is "feature"',
        required=False,
        default='feature'
    )

    parser.add_argument(
        '-e', '--epochs',
        help='Set number of items for removal, default 800',
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
        cmd_line_args.list_datasets,
        cmd_line_args.operation,
        cmd_line_args.dataset,
        cmd_line_args.nametag,
        cmd_line_args.unlearn,
        cmd_line_args.display_metrics,
        cmd_line_args.display_metrics_nametags,
        cmd_line_args.display_inference_score,
        cmd_line_args.epochs,
        cmd_line_args.num_removes,
        cmd_line_args.removal_mode,
        cmd_line_args.verbose,
    )


if __name__ == '__main__':
    main()
