# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.3  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 3e-5      # Initial learning rate.


BATCH_SIZE = 10
WEIGHT_DECAY = 0.001
MAX_STEPS = 10000
