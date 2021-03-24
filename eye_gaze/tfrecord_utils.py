import tensorflow as tf
from functools import partial

BATCH_SIZE = 64
IMAGE_SIZE = [32, 32]
AUTOTUNE = tf.data.experimental.AUTOTUNE

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, IMAGE_SIZE)
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


def load_dataset(filename, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filename
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def get_dataset(filename, labeled=True):
    dataset = load_dataset(filename, labeled=labeled)
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