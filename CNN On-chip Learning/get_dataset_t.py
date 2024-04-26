import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical

def build_dataset(which_dataset='cifar10'):
    '''
    Build and preprocess a dataset for training and testing.

    Parameters:
    - which_dataset: Name of the dataset to build ('cifar10', 'mnist', or 'fashion_mnist').

    Returns:
    - dataset: Training dataset.
    - dataset_test: Testing dataset.
    '''
    
    # Dictionary mapping dataset names to their respective sources
    src_dict = {
        'cifar10': cifar10,
        'mnist': mnist, 
        'fashion_mnist': fashion_mnist
    }

    # Load the specified dataset
    src = src_dict[which_dataset] 
    (x_train, y_train_), (x_test, y_test_) = src.load_data()
    
    # Preprocess the data
    x_train = 2 * x_train.astype('float32') / 255 - 1
    x_test = 2 * x_test.astype('float32') / 255 - 1
    y_train = to_categorical(y_train_)
    y_test = to_categorical(y_test_)

    # If the dataset is not 'cifar10', expand dimensions to account for channels
    if which_dataset != 'cifar10':
        x_train = tf.expand_dims(x_train, -1)
        x_test = tf.expand_dims(x_test, -1)

    # Create training dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_))
    dataset = dataset.shuffle(buffer_size=1024).batch(1)

    # Create testing dataset
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test_))
    dataset_test = dataset_test.shuffle(buffer_size=1024).batch(1)

    return dataset, dataset_test
