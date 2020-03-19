# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import losses
import utils
import os
import argparse
from functools import reduce
from operator import mul

slim = tf.contrib.slim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/mosaic.yml', help='the path to the conf file')
    parser.add_argument('-lc','--list_config',default=None,help='the path of the conf file list')
    return parser.parse_args()


def main(FLAGS):

    #done training path exists
    #训练完毕时的库路径
    training_done_path = os.path.join('TrainingDone/{0}'.format(FLAGS.model_path), FLAGS.naming)
    if not (os.path.exists(training_done_path)):
        os.makedirs(training_done_path)

    #获取样式的特征tensor
    #这里将需要学习的样式图片，和样式网络怼起来，输出的是VGG 部分网络的结果
    style_features_t = losses.get_style_features(FLAGS)

    # Make sure the training path exists.
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)

    if not (os.path.exists(training_path)):
        os.makedirs(training_path)


    with tf.Graph().as_default():
        with tf.Session() as sess:
            """Build Network"""
            #获取vgg Network，必须不用重新训练权重参数
            network_fn = nets_factory.get_network_fn(
                FLAGS.loss_model,
                num_classes=1,
                is_training=False)

            #获取图片的预处理和后处理程序
            #image_preprocessing_fn 会将输入的图片，进行裁剪，并且对每个通道都减去一个平均值(这个平均值是写死的，经验值)
            #image_unpreprocessing_fn 会将输入的图片的每个通道，都加上这个平均值(这个平均值同上)
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            #处理输入的图片，经过image_preprocessing_fn函数指针
            #输入train2014文件夹里面所有的内容，按照打乱的方式，对每个图片进行预处理之后，输出
            #processed_images = [4,256,256,3]
            processed_images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size,
                                            'train2014/', image_preprocessing_fn, epochs=FLAGS.epoch)
            #获取transfer的模型输出， 若干个卷积层级，包括残差，以及padding操作
            #图片的网络输出
            #generated = [4,256,256,3]
            generated = model.net(processed_images, training=True)
            #对每一张输出的图片进行预处理操作，tf.unstack就是对tensor进行拆解
            #processed_generated =list([256,256,3])，维度为batch_size
            processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                                   for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)
                                   ]

            # 新增一维处理之后的图片
            #processed_generated = [4,256,256,3]
            processed_generated = tf.stack(processed_generated)  #多加了一维processed_generated
            # 获取对应的VGG模型每一层的输出节点
            # tf.concat 在第0维的位置，拼接数据， processed_images 输入， processed_generated为经过风格转换网络之后的输出
            # network_fn 是一开始就通过factory加载的VGG网络
            # 在第0维加入维度,包含预处理的输入图片和经过transfer模型输出的图片
            #tf.concat([processed_generated, processed_images],0) = [8,256,256,3]
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False) #

            # Log the structure of loss network
            tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
            #打印每一层的网络
            for key in endpoints_dict:
                tf.logging.info(key)

            """Build Losses"""
            #计算content_loss , endpoints_dict 里面是tensor的合集，命名是以layer名称
            content_loss = losses.content_loss(endpoints_dict, FLAGS.content_layers)
            #计算Style_loss，style_features_t也是一个tensor，真实数据合集
            style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, FLAGS.style_layers)
            #计算Total loss
            tv_loss = losses.total_variation_loss(generated)  # use the unprocessed image
            #计算总的loss
            loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * tv_loss

            # Add Summary for visualization in tensorboard.
            """Add Summary"""
            tf.summary.scalar('losses/content_loss', content_loss)  # 记录监控数据，首个参数命名为content_loss
            tf.summary.scalar('losses/style_loss', style_loss)  # 同上
            tf.summary.scalar('losses/regularizer_loss', tv_loss)  # 同上

            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * FLAGS.content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * FLAGS.style_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * FLAGS.tv_weight)
            tf.summary.scalar('total_loss', loss)

            for layer in FLAGS.style_layers:
                tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            tf.summary.image('generated', generated)
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.summary.image('origin', tf.stack([
                image_unprocessing_fn(image) for image in tf.unstack(processed_images, axis=0, num=FLAGS.batch_size)
            ]))
            summary = tf.summary.merge_all()  #merge_all 可以汇总所有的信息
            writer = tf.summary.FileWriter(training_path) #写入这个Path

            """Prepare to Train"""
            #准备开始训练了
            global_step = tf.Variable(0, name="global_step", trainable=False)
            
            variable_to_train = []
            for variable in tf.trainable_variables():
                #排除掉vgg开头的变量名称
                if not(variable.name.startswith(FLAGS.loss_model)):
                    variable_to_train.append(variable)
            #s输入优化器，学习率为0.001
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variables_to_restore = []
            for v in tf.global_variables():
                if not(v.name.startswith(FLAGS.loss_model)):
                    variables_to_restore.append(v)
            #新建saver，是不需要重新训练的参数
            saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)
            #图表已经搭建完成，现在开始训练
            #可以开始run了
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Restore variables for loss network.
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)  #对权重进行赋值(此时还没有赋值，只是用sess初始化了赋值函数)

            # Restore variables for training model if the checkpoint file exists.
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)  # 这里才开始真正的赋值

            """Start Training"""
            #新建线程管理器
            coord = tf.train.Coordinator()
            #获取运行的线程
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()

            print("正在训练的参数量为 %d"%get_num_params())

            try:
                #线程管理还未终止
                while not coord.should_stop():
                    #计算---训练， 这一步之后，才会有对应的结果输出
                    #train_op是优化器的输出，loss是总的样式损耗，global_step是tf变量
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    # print(step)
                    if step % 10 == 0:
                        tf.logging.info('config file:%s, step: %d,  total Loss %f, secs/step: %f' %
                                        (FLAGS.naming,step, loss_t, elapsed_time))
                        print('config file:%s,step: %d,  total Loss %f, secs/step: %f' % (
                        FLAGS.naming, step, loss_t, elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        # 每一千步保存checkpoint
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
            except tf.errors.OutOfRangeError:

                saver.save(sess, os.path.join(training_done_path, 'fast-style-model.ckpt-done'))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                #线程管理要求停止
                coord.request_stop()
            #等待 threads结束
            coord.join(threads)


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()

    if args.list_config is not None:  #输入参数是否定义了list_config
        config_list = utils.read_conf_file(args.list_config) #读取配置文件
        for config_file in config_list.trainning_list:
            FLAGS = utils.read_conf_file(config_file)
            main(FLAGS)
    else:   # trainnig single file
        FLAGS = utils.read_conf_file(args.conf)
        main(FLAGS)



