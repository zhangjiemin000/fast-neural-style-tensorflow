# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import utils

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                                                   'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
# tf.app.flags.DEFINE_string("config_file", "models.ckpt", "")
# tf.app.flags.DEFINE_string("image_file", "./img/timg.jpg", "")
tf.app.flags.DEFINE_string("config_file","conf/youhua2.yml","")

FLAGS = tf.app.flags.FLAGS



def process_img(saver,sess,generated,model_file_name,img_name):
    model_file_path = os.path.abspath(os.path.join('TrainingDone/models',model_file_name,'fast-style-model.ckpt-done'))
    if(not os.path.exists(model_file_path)):
        return
    # Use absolute path
    # FLAGS.model_file = os.path.abspath(FLAGS.model_file)
    saver.restore(sess, model_file_path)
    generated_file = os.path.join('../fast-neural-test/generated',img_name+'_'+model_file_path.split("/")[-2]+'.jpg')
    # Make sure 'generated' directory exists.
    # generated_file = 'generated/res.jpg'
    if os.path.exists('generated') is False:
        os.makedirs('generated')

    # Generate and write image data to file.
    with open(generated_file, 'wb') as img:
        start_time = time.time()
        sss = tf.image.encode_jpeg(generated)
        img.write(sess.run(tf.image.encode_jpeg(generated)))
        end_time = time.time()
        tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
        tf.logging.info('Done. Please check %s.' % generated_file)



def main(_):
    test_imgs_list = []
    filter = ['.jpg','.jpeg','.png']
    for maindir, subdir, file_name_list in os.walk('../fast-neural-test/testPics/'):
        for file_name in file_name_list:
            apath = os.path.join(maindir,file_name)
            ext = os.path.splitext(apath)[1]
            if ext in filter:
                test_imgs_list.append(apath)
    for image_file in test_imgs_list:
        img_name = os.path.basename(image_file)
        img_name = os.path.splitext(img_name)[0]
        with open(image_file, 'rb') as img:
            with tf.Session().as_default() as sess:
                if image_file.lower().endswith('png'):
                    image = sess.run(tf.image.decode_png(img.read()))
                else:
                    image = sess.run(tf.image.decode_jpeg(img.read()))
                height = image.shape[0]
                width = image.shape[1]
        tf.logging.info('Image size: %dx%d' % (width, height))

        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:

                # Read image data.
                image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                    FLAGS.loss_model,
                    is_training=False)
                image = reader.get_image(image_file, height, width, image_preprocessing_fn)

                # Add batch dimension
                image = tf.expand_dims(image, 0)

                generated = model.net(image, training=False)
                generated = tf.cast(generated, tf.uint8)

                # Remove batch dimension
                generated = tf.squeeze(generated, [0])

                # Restore model variables.
                saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                config = utils.read_conf_file(FLAGS.config_file)
                for weight in range(config.min_style_weight, config.max_style_weight, config.style_weigth_step):
                    config.style_weight = weight
                    name = config.naming+"_"+str(config.style_weight)
                    process_img(saver,sess,generated,name,img_name)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
