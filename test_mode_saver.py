import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

with tf.variable_scope('Placeholder'):
    inputs_x = tf.placeholder(tf.float32, name='inputs_x', shape=[None, 10])
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='labels_placeholder')
with tf.variable_scope('NN'):
    w1 = tf.get_variable(name='w1', shape=[10,1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b1 = tf.get_variable(name='b1', shape=[1], initializer=tf.constant_initializer(0.1))
    w2 = tf.get_variable('w2', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))

    a = tf.nn.relu(tf.matmul(inputs_x, w1)+b1)
    a2 = tf.nn.relu(tf.matmul(inputs_x, w2)+b2)

    y = tf.div(tf.add(a, a2), 2)
with tf.variable_scope('loss'):
    loss = tf.reduce_sum(tf.square(y-labels_placeholder) / 2)
with tf.variable_scope('Accuracy'):
    predictions = tf.greater(y , 0.5, name='predictions')
    correct_predictions = tf.equal(predictions, tf.cast(labels_placeholder, dtype=tf.bool), name='correct_preditions')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
with tf.variable_scope('train'):
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

#generate data
inputs = np.random.choice(10, size=[10000, 10]) #np.arange(10)
labels = (np.sum(inputs, axis=1) > 45).reshape(-1, 1).astype(np.float32)
print('inputs.shape: ', inputs.shape)
print('labels.shape: ', labels.shape)

test_inputs = np.random.choice(10, size=[100, 10]) #np.arange(10)
test_labels = (np.sum(test_inputs, axis=1) > 45).reshape(-1, 1).astype(np.float32)
print('test_inputs.shape: ', test_inputs.shape)
print('test_labels.shape: ', test_labels.shape)

batch_size = 32
epochs = 10
batches = []

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

batchs = batch_iter(list(zip(inputs, labels)), batch_size, epochs)

saver = tf.train.Saver(max_to_keep=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch in batchs:
        input_batch, label_batch = zip(*batch)
        train_loss, _ = sess.run([loss, train_op], feed_dict={inputs_x:input_batch,labels_placeholder:label_batch})
        global_count = sess.run(global_step)
        if global_count % 100 == 0:
            last_chkp = saver.save(sess, 'results/graph.chkp', global_step=global_count)
            print('step: %d, train_loss: %d' % (global_count,train_loss))
            acc = sess.run(accuracy, feed_dict={inputs_x:test_inputs,
                                                labels_placeholder:test_labels})
            print('step: %d, accuracy: %f'% (global_count, acc))

    acc = sess.run(accuracy, feed_dict={
        inputs_x: test_inputs,
        labels_placeholder: test_labels })
    print("final accuracy: %f" % acc)

# for op in tf.get_default_graph().get_operations():
#     print(op.name)