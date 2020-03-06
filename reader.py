from os import listdir
from os.path import isfile, join
import tensorflow as tf


def get_image(path, height, width, preprocess_fn):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)


def image(batch_size, height, width, path, preprocess_fn, epochs=2, shuffle=True):
    # 列举所有的文件名， 作为列表
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    #如果不要打乱顺序，则按文件名排名
    if not shuffle:
        filenames = sorted(filenames)

    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are

    #对文件列表进行打乱(其实就是对文本进行打乱操作)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    #这个可以读入非常多的文件
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    #decode 所有的image
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    #预处理图片(将三个通道分别加减数值)
    processed_image = preprocess_fn(image, height, width)
    #按照batchSize来分配图片
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)
