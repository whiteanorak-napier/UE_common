
from datetime import datetime
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
UE_UNLEARN_MODEL = 'unlearn'
UE_TRAIN_MODEL = 'train'

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
            if not isinstance(value, float):
                print(f"Loss score must be a float. Got '{value}")
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
            if not isinstance(value, float):
                print(f"Loss score must be a float. Got '{value}")
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

def ue_store_model(model, filename, train_or_unlearn=UE_TRAIN_MODEL):
    if filename is not None:
        if train_or_unlearn == UE_UNLEARN_MODEL:
            filename_split = filename.split('.')
            filename = filename_split[0] + '_unlearn' + '.pt'
        torch.save(model, filename)

def extract_token_value(message, token):
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
