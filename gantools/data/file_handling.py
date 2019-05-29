import numpy as np

from ast import literal_eval as make_tuple
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes

def read_tfrecords_from_file(file_path, image_size, k=10.):
    '''
    read samples stored in a tfrecord file
    '''
    record_iterator = tf.python_io.tf_record_iterator(path=file_path)
    cubes = []
    for string_record in record_iterator:

        # Parse each record out from the tf records file
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])

        cube = np.fromstring(img_string, dtype=np.float32)
        cube = cube.reshape(image_size)
        cubes.append(cube)

    data = np.array(cubes, dtype=np.float32)    
    forward_mapped_data = utils.forward_map(data, k)

    return forward_mapped_data, data

def read_tfrecords_from_dir(dir_path, image_size, k):
    '''
    read samples from all tfrecord files in a directory
    '''
    vstacked_forward = 0
    vstacked_raw = 0
    first = True

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        forward_mapped_data, data = read_tfrecords_from_file(file_path, image_size, k)
        if first:
            first = False
            vstacked_forward = forward_mapped_data
            vstacked_raw = data
        else:
            vstacked_forward = np.vstack((vstacked_forward, forward_mapped_data))
            vstacked_raw = np.vstack((vstacked_raw, data))

    return vstacked_forward, vstacked_raw


## Create input pipeline for reading in input from files

def create_input_pipeline(dir_paths, batch_size, k=10, shuffle=False, buffer_size=1000):
    sample_file_paths, num_samples, sample_dims = read_sample_file_paths(dir_paths)

    if shuffle:
        random.shuffle(sample_file_paths)

    dataset = tf.data.TFRecordDataset(sample_file_paths)

    def parser(serialized_example):
        """Parses a single tf.Example into image"""
        parsed_features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string)
            })
        
        image = parsed_features['image']
        image = tf.decode_raw(image, tf.float32)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, sample_dims)
        image = utils.forward_map(image, k) # forward map the raw images
        return image

    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset, num_samples

def read_sample_file_paths(dir_paths):
    '''
    store paths of all sample files in a list
    '''
    num_samples = 0
    sample_dims = []
    sample_file_paths = []
    for dir_path in dir_paths:
        for file_name in os.listdir(dir_path):
            parts = file_name.split('_')
            num_samples += int(parts[1])
            dims = parts[2].split('.')[0]
            sample_dims = make_tuple(dims)

            file_path = os.path.join(dir_path, file_name)
            sample_file_paths.append(file_path)

    return sample_file_paths, num_samples, sample_dims