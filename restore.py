def restoreChkPoint():
    random_dim = 100

    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    print(".....fake_image pass")
    real_result = discriminator(real_image, is_train)
    print(".....real_pass pass")
    fake_result = discriminator(fake_image, is_train, reuse=True)
    print(".....fake_res pass")

    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    #print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, ckpt)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

    #d_loss = tf.reduce_mean(tf.log(real_result)) - tf.log(1 - fake_result)  # This optimizes the discriminator.
    #g_loss = -tf.reduce_mean(tf.log(fake_result))  # This optimizes the generator.


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]


    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')

    start_time = time.time()

    for i in range(EPOCH):
        epoch_time = time.time()
        text_file = open("Output.txt", "a+")
        print("Running epoch {}/{}...".format(i, EPOCH))
        for j in range(batch_num):
            print(j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print(k)
                #with tf.device('/gpu:1'):
                train_image = sess.run(image_batch)
                #wgan clip weights
                #with tf.device('/gpu:0'):
                sess.run(d_clip)

                # Update the discriminator
                #with tf.device('/gpu:1'):
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                #with tf.device('/gpu:2'):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            print ('train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss))
        print("--- EPOCH time : %s seconds ---" % (time.time() - epoch_time))


        # save check point every 500 epoch
        if i%500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))
        if i%250 == 0:
            # save images
            if not os.path.exists(newSlab_path):
                os.makedirs(newSlab_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [6,6] ,newSlab_path + '/epoch' + str(i) + '.jpg')
        if i%10 == 0:
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
            print("--- total time : %s seconds ---" % (time.time() - start_time))
            text_file.write("%d \t\t%f \t\t%f \t\t%s \t\t%s \n" % (i, dLoss, gLoss, (time.time() - epoch_time),(time.time() - start_time)))
            text_file.close()




    #text_file.close()
    coord.request_stop()
    coord.join(threads)
