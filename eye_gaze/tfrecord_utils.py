import tensorflow as tf
from functools import partial
import numpy as np

BATCH_SIZE = 64
IMAGE_SIZE = [32, 32]
AUTOTUNE = tf.data.experimental.AUTOTUNE

def decode_image(image):
    img = np.frombuffer(image.numpy(), dtype='B')
    img = img.reshape(32,32)  # dimensions of the image
    return img.astype(np.uint8)


def read_tfrecord(example):
    tfrecord_format = (
        {
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'center': tf.io.VarLenFeature(tf.float32),
            'inner': tf.io.VarLenFeature(tf.float32),      
            'outer': tf.io.VarLenFeature(tf.float32),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format, name='features')                                                                                                                                           
    image = example["image"]
    center = example['center']
    inner = example['inner']
    outer = example['outer']

    return image, center, inner, outer


def load_dataset(filename):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filename
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def get_dataset(filename):
    dataset = load_dataset(filename)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

#=======================================================================#

def landmark_feature(value):
    value = list(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def image_feature(value):
    value = [value.tobytes()]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def generate_examples(data):
    examples = []
    if('croppedLeft' in data):
        example_left = tf.train.Example(features=tf.train.Features(feature={
            'image': image_feature(data['croppedLeft']),
            'center': landmark_feature(data['cl_center']),
            'inner': landmark_feature(data['cl_inner_corner']),      
            'outer': landmark_feature(data['cl_outer_corner']),
        }))
        examples.append(example_left)

    if('croppedRight' in data):
        example_right = tf.train.Example(features=tf.train.Features(feature={
            'image': image_feature(data['croppedRight']),
            'center': landmark_feature(data['cr_center']),
            'inner': landmark_feature(data['cr_inner_corner']),    
            'outer': landmark_feature(data['cr_outer_corner']),
        }))
        examples.append(example_right)
    return examples

def write_tfrecord(examples, writer):
    for example in examples:
        writer.write(example.SerializeToString())


#==============================================================#
"""
    tfrecord_format = (
        {
            'image': tf.io.FixedLenFeature([], tf.string),
            #'center': tf.io.FixedLenFeature([], tf.float32),
            #'inner': tf.io.FixedLenFeature([], tf.float32),      
            #'outer': tf.io.FixedLenFeature([], tf.float32),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format, name='features')
    image = example["image"]
    
    landmarks = {
        'center': example['center'],
        'inner' : example['inner'],
        'outer': example['outer'],
    }

"""