# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import utils
import os

slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters])) #重组shape，tf.stack用于更改形状特征的，这里的Stack的意义是:[1,-1,256],组成这个
    #求解格拉姆矩阵值
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)
        #获取 image 预处理方法指针，和image 反处理方法指针
        #如是vgg的模型，那就取 vggprocessing 类里面的 process 和 unprocessing 的方法
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)

        # Get the style image data
        #获取Style Image 的Size
        size = FLAGS.image_size  # 获取StyleImage的Size
        #读取style image 的内容
        img_bytes = tf.read_file(FLAGS.style_image)  # Tensor
        #解码，jpg和png
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)  #解析图片
        else:
            image = tf.image.decode_jpeg(img_bytes)  #解析图片 Tensor
        # image = _aspect_preserving_resize(image, size)

        # Add the batch dimension
        # 把style_image的处理成接近于flag中设置的Size大小，先等比例缩小到适当的大小，再居中裁剪掉超出的部分
        #shape=[1,]
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        # images = tf.stack([image_preprocessing_fn(image, size, size)])
        #将样式图片作为输入，进入到对应的VGG网络中
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        #获取 Style_layers
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)

        #准备工作完成后，开始准备进入真正的训练阶段了
        with tf.Session() as sess:
            # Restore variables for loss network.
            # 还原VGG的权重参数，从对应的Vgg checkpoint文件中
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess) #把session作为参数传入

            # Make sure the 'generated' directory is exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            # Indicate cropped style image path
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                # 存储样式图片到本地
                target_image = image_unprocessing_fn(images[0, :]) #把样式image进行后处理
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # Return the features those layers are use for measuring style loss.
            return sess.run(features)


def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}
    # zip用来将输入的参数打包成一个元祖进行遍历，第一个参数的一个元素和第二个参数的第一个元素组合
    for style_gram, layer in zip(style_features_t, style_layers):
        # 找出第0轴，按照size=2来分割这个shape， style_gram和layer就是style_features_t 和 style_layers这两个里面的参数
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss  # 加上所有的style_loss
    return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    return content_loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    #这里将layer 挪一个位置 相减 获得x、y的值，然后再做计算得到loss
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
