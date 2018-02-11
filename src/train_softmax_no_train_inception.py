from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
from datetime import datetime

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.slim as slim

from loss.sphere import *
from models import inception_resnet_no_train as network
from utils import faceUtil


def main(args):
    print("main start")
    np.random.seed(seed=args.seed)
    #train_set =   ImageClass list
    train_set = faceUtil.get_dataset(args.data_dir)

    #总类别
    nrof_classes = len(train_set)
    print(nrof_classes)

    #subdir =20171122-112109
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    #log_dir = c:\User\logs\facenet\20171122-
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir),subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print("log_dir =",log_dir)

    # model_dir =c:\User/models/facenet/2017;;;
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    print("model_dir =", model_dir)
    pretrained_model = None
    if args.pretrained_model:
        # pretrained_model = os.path.expanduser(args.pretrained_model)
        # pretrained_model = tf.train.get_checkpoint_state(args.pretrained_model)
        pretrained_model = args.pretrained_model
        print('Pre-trained model: %s' % pretrained_model)


    # Write arguments to a text file
    faceUtil.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    print("write_arguments_to_file")
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0,trainable=False)

        #两个列表 image_list= 图片地址列表, label_list = 对应label列表，两个大小相同
        image_list, label_list  = faceUtil.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0 , 'dataset is empty'
        print("len(image_list) = ",len(image_list))

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list,dtype=tf.int64)
        range_size = array_ops.shape(labels)[0]
        range_size = tf.Print(range_size, [tf.shape(range_size)],message='Shape of range_input_producer range_size : ',summarize=4, first_n=1)

        #产生一个队列，队列包含0到range_size-1的元素,打乱
        index_queue = tf.train.range_input_producer(range_size,num_epochs=None,shuffle=True,seed=None,capacity=32)

        #从index_queue中取出 args.batch_size*args.epoch_size  个元素，用来从image_list, label_list中取出一部分feed给网络
        index_dequeue_op = index_queue.dequeue_many(args.batch_size *  args.epoch_size,'index_dequeue')

        #学习率
        learning_rate_placeholder = tf.placeholder(tf.float32,name='learning_rate')
        #批大小 arg.batch_size
        batch_size_placeholder = tf.placeholder(tf.int32,name='batch_size')
        #是否训练中
        phase_train_placeholder = tf.placeholder(tf.bool,name='phase_train')
        #图像路径 大小 arg.batch_size * arg.epoch_size
        image_paths_placeholder = tf.placeholder(tf.string,shape=[None,1],name='image_paths')
        #图像标签 大小：arg.batch_size * arg.epoch_size
        labels_placeholder = tf.placeholder(tf.int64,shape=[None,1],name='labels')

        #新建一个队列,数据流操作,fifo,先入先出
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,dtypes=[tf.string,tf.int64],shapes=[(1,),(1,)],shared_name=None,name=None)

        # enqueue_many返回的是一个操作 ,入站的数量是 len（image_paths_placeholder) = 从index_queue中取出 args.batch_size*args.epoch_size个元素
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder,labels_placeholder],name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []

        for _ in range(nrof_preprocess_threads):
            filenames , label = input_queue.dequeue()
            # label = tf.Print(label,[tf.shape(label)],message='Shape of one thread  input_queue.dequeue label : ',
            #                  summarize=4,first_n=1)
            # filenames = tf.Print(filenames, [tf.shape(filenames)], message='Shape of one thread  input_queue.dequeue filenames : ',
            #                  summarize=4, first_n=1)
            print("one thread  input_queue.dequeue len = ",tf.shape(label))
            images =[]
            for filenames in tf.unstack(filenames):
                file_contents = tf.read_file(filenames)
                image = tf.image.decode_image(file_contents,channels=3)

                if args.random_rotate:
                    image = tf.py_func(faceUtil.random_rotate_image, [image], tf.uint8)

                if args.random_crop:
                    image = tf.random_crop(image,[args.image_size,args.image_size,3])

                else:
                    image = tf.image.resize_image_with_crop_or_pad(image,args.image_size,args.image_size)

                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                image.set_shape((args.image_size,args.image_size,3))
                images.append(tf.image.per_image_standardization(image))

            #从队列中取出名字 解析为image 然后加进images_and_labels 可能长度 =  4 *
            images_and_labels.append([images,label])

        #最终一次进入网络的数据: 长应该度 = batch_size_placeholder
        image_batch, label_batch = tf.train.batch_join(images_and_labels,batch_size=batch_size_placeholder,
                                                       shapes=[(args.image_size,args.image_size,3),()],
                                                       enqueue_many = True,
                                                       capacity = 4 * nrof_preprocess_threads *  args.batch_size,
                                                       allow_smaller_final_batch=True)
        print('final input net  image_batch len = ',tf.shape(image_batch))

        image_batch = tf.Print(image_batch, [tf.shape(image_batch)], message='final input net  image_batch shape = ',
                         summarize=4, first_n=1)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # 将指数衰减应用到学习率上
        learning_rate = tf.train.exponential_decay(learning_rate= learning_rate_placeholder,
                                                   global_step = global_step,
                                                   decay_steps=args.learning_rate_decay_epochs * args.epoch_size,
                                                   decay_rate=args.learning_rate_decay_factor,
                                                   staircase = True)
        #decay_steps=args.learning_rate_decay_epochs * args.epoch_size,

        tf.summary.scalar('learning_rate', learning_rate)

        # Build the inference graph
        prelogits, _ = network.inference(image_batch,args.keep_probability,phase_train=phase_train_placeholder,
                                         bottleneck_layer_size=args.embedding_size,weight_decay=args.weight_decay)

        prelogits = tf.Print(prelogits, [tf.shape(prelogits)], message='prelogits shape = ',
                               summarize=4, first_n=1)
        print("prelogits.shape = ",prelogits.get_shape().as_list())

        # logits =slim.fully_connected(prelogits, len(train_set), activation_fn=None,
        #                               weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                               weights_regularizer=slim.l2_regularizer(args.weight_decay),
        #                               scope='Logits', reuse=False)
        #
        # # Calculate the average cross entropy loss across the batch
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=label_batch, logits=logits, name='cross_entropy_per_example')
        # tf.reduce_mean(cross_entropy, name='cross_entropy')
        _,cross_entropy_mean = soft_loss(prelogits,label_batch,len(train_set))
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses,name='total_loss')

        train_op = faceUtil.train(total_loss, global_step, args.optimizer, learning_rate,
                                  args.moving_average_decay, tf.trainable_variables(), args.log_histograms)
        # print("global_variables len = {}".format(len(tf.global_variables())))
        # print("local_variables len = {}".format(len(tf.local_variables())))
        # print("trainable_variables len = {}".format(len(tf.trainable_variables())))
        # for v in tf.trainable_variables() :
        #     print("trainable_variables :{}".format(v.name))
        # train_op = faceUtil.train(sphere_loss,global_step,args.optimizer,learning_rate,
        #                   args.moving_average_decay, tf.global_variables(), args.log_histograms)

        #创建saver
        variables = tf.trainable_variables()
        print("variables_trainable len = ", len(variables))
        # for v in variables:
        #      print('variables_trainable : {}'.format(v.name))
        saver = tf.train.Saver(var_list=variables, max_to_keep=2)

        variables_to_restore = slim.get_variables_to_restore(include=['InceptionResnetV1'])

        print("variables_to_restore len = ", len(variables_to_restore))
        saver_restore = tf.train.Saver(var_list=variables_to_restore)

        # variables_to_restore  = [v for v in variables if v.name.split('/')[0] != 'Logits']
        # print("variables_trainable len = ",len(variables))
        # print("variables_to_restore len = ",len(variables_to_restore))
        # # for v in variables_to_restore :
        # #     print("variables_to_restore : ",v.name)
        # saver = tf.train.Saver(var_list=variables_to_restore,max_to_keep=3)


        # variables_trainable = tf.trainable_variables()
        # print("variables_trainable len = ",len(variables_trainable))
        # # for v in variables_trainable :
        # #     print('variables_trainable : {}'.format(v.name))
        # variables_to_restore = slim.get_variables_to_restore(include=['InceptionResnetV1'])
        # print("variables_to_restore len = ",len(variables_to_restore))
        # saver = tf.train.Saver(var_list=variables_to_restore,max_to_keep=3)



        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # 能够在gpu上分配的最大内存
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options,log_device_placement = False))

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # 获取线程坐标
        coord = tf.train.Coordinator()

        # 将队列中的所有Runner开始执行
        tf.train.start_queue_runners(coord=coord,sess=sess)

        with sess.as_default():
            print('Running training')
            if pretrained_model :
                print('Restoring pretrained model_checkpoint_path: %s' % pretrained_model)
                saver_restore.restore(sess,pretrained_model)

            # Training and validation loop
            print('Running training really')
            epoch = 0
            # 将所有数据过一遍的次数
            while epoch < args.max_nrof_epochs:

                #这里是返回当前的global_step值吗,step可以看做是全局的批处理个数
                step = sess.run(global_step,feed_dict=None)

                #epoch_size是一个epoch中批的个数
                # 这个epoch是全局的批处理个数除以一个epoch中批的个数得到epoch,这个epoch将用于求学习率
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir



def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):

        batch_number = 0
        if args.learning_rate>0.0:

            lr = args.learning_rate
        else:
            lr = faceUtil.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

        index_epoch = sess.run(index_dequeue_op)
        label_epoch = np.array(label_list)[index_epoch]
        image_epoch = np.array(image_list)[index_epoch]

        # Enqueue one epoch of image paths and labels
        labels_array = np.expand_dims(np.array(label_epoch),1)
        image_paths_array = np.expand_dims(np.array(image_epoch),1)
        sess.run(enqueue_op,{image_paths_placeholder:image_paths_array,labels_placeholder:labels_array})

        # Training loop
        train_time = 0
        while batch_number < args.epoch_size:

            start_time = time.time()
            feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}

            if (batch_number % 100 == 0) :
                err, _, step, reg_loss, summary_str = sess.run([loss,train_op,global_step,regularization_losses,summary_op],feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
            else :
                err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)

            duration = time.time() - start_time
            print('global_step[%d],Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (step,epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
            batch_number += 1
            train_time += duration

        # Add validation loss and accuracy to summary
        summary = tf.Summary()

        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/total', simple_value=train_time)
        summary_writer.add_summary(summary, step)
        return step



def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):

    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir,'model-%s.ckpt' % model_name)
    saver.save(sess,checkpoint_path,global_step=step,write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir,'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)

    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)





















def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    print("parser = argparse.ArgumentParser()")
    parser.add_argument('--data_dir',type=str,default='align/casia_maxpy_mtcnnpy_182')
    parser.add_argument('--gpu_memory_fraction',type=float,default=0.8)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    default = '/home/huyu/models/facenet/20180209-114624/model-20180209-114624.ckpt-0',
    #default='/home/huyu/models/facenet/20180208-210946/model-20180208-210946.ckpt-1200',
    # default='modeltrained/20170512/model-20170512-110547.ckpt-250000',

    parser.add_argument('--max_nrof_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size',type=int,default=160)
    parser.add_argument('--epoch_size', type=int, default=300)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--random_crop',
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip',
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=1e-8)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.05)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=1)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.9)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_classifier_casia.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
