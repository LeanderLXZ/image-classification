import pickle
import helper
import tensorflow as tf
import math


# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


# Hyperparameters #

epochs = 100
batch_size = 64
keep_probability = 0.75

conv1_ksize = 3
conv1_strides = 1
conv1_depth = 64

conv2_ksize = 3
conv2_strides = 1
conv2_depth = 128

conv3_ksize = 3
conv3_strides = 1
conv3_depth = 256

fc1_nodes = 512
fc2_nodes = 256
fc3_nodes = 128

version = 'V1.0'

save_model_path = './saver/full_batch/' + version

# Input #

def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    x = tf.placeholder(tf.float32, [None, *image_shape], name='x')
    return x


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')
    return y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return keep_prob


# Convolution and Max Pooling Layer #

def conv2d_maxpool(x_tensor, layer_name, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    with tf.name_scope(layer_name):
        x_shape = x_tensor.get_shape().as_list()

        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([*conv_ksize,
                                                    x_shape[3],
                                                    conv_num_outputs], stddev=0.1))
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([conv_num_outputs]))
            variable_summaries(biases)

        conv = tf.nn.conv2d(x_tensor,
                            weights,
                            strides=[1, *conv_strides, 1],
                            padding='SAME')

        conv = tf.nn.elu(tf.nn.bias_add(conv, biases))

        max_pool = tf.nn.max_pool(conv,
                                  ksize=[1, *pool_ksize, 1],
                                  strides=[1, *pool_strides, 1],
                                  padding='SAME')

    return max_pool


# Flatten Layer #

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    with tf.name_scope('flatten'):
        shape = x_tensor.get_shape().as_list()
        flatten_num = shape[1] * shape[2] * shape[3]
        flatten = tf.reshape(x_tensor, [-1, flatten_num])
    return flatten


# Fully-Connected Layer #

def fully_conn(x_tensor, layer_name, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    with tf.name_scope(layer_name):
        x_shape = x_tensor.get_shape().as_list()

        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([num_outputs]))
            variable_summaries(biases)

        with tf.name_scope('Wx_plus_b'):
            fc_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            tf.summary.histogram('fc_layer', fc_layer)

        logits = tf.nn.elu(fc_layer)
        tf.summary.histogram('logits', logits)

    return logits


# Output Layer #

def output(x_tensor, layer_name, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    with tf.name_scope(layer_name):
        x_shape = x_tensor.get_shape().as_list()

        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([num_outputs]))
            variable_summaries(biases)

        with tf.name_scope('Wx_plus_b'):
            output_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            tf.summary.histogram('output', output_layer)

    return output_layer


# Create Convolutional Model #

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    #  Convolutional layers
    conv1 = conv2d_maxpool(x, 'conv_layer1', conv1_depth, (conv1_ksize, conv1_ksize), (conv1_strides, conv1_strides), (2, 2), (2, 2))
    conv2 = conv2d_maxpool(conv1, 'conv_layer2', conv2_depth, (conv2_ksize, conv2_ksize), (conv2_strides, conv2_strides), (2, 2), (2, 2))
    conv3 = conv2d_maxpool(conv2, 'conv_layer3', conv3_depth, (conv3_ksize, conv3_ksize), (conv3_strides, conv3_strides), (2, 2), (2, 2))

    # Flatten layer
    flat = flatten(conv3)

    #  Full connected layers
    fc1 = fully_conn(flat, 'fc_layer1', fc1_nodes)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = fully_conn(fc1, 'fc_layer2', fc2_nodes)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    # fc3 = fully_conn(fc2, 'fc_layer3', fc3_nodes)
    # fc3 = tf.nn.dropout(fc3, keep_prob)

    # Output layer
    out = output(fc2, 'output_layer', 10)

    return out


# Build the Neural Network #

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
tf.summary.scalar('cost', cost)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)


# Train the Neural Network #

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x: feature_batch,
                                      y: label_batch,
                                      keep_prob: keep_probability,})


# Train on a Single CIFAR-10 Batch #

print('Checking the Training on a Single Batch...')

with tf.Session() as sess:

    # Merge all the summaries and write them
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/single_batch/' + version + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./log/single_batch/' + version + '/test')

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')

        summary_train, cost_stats = sess.run([merged, cost],
                                             feed_dict={x: batch_features,
                                                        y: batch_labels,
                                                        keep_prob: 1.0},)
        train_writer.add_summary(summary_train, epoch)

        summary_test, valid_accuracy = sess.run([merged, accuracy],
                                                feed_dict={x: valid_features,
                                                           y: valid_labels,
                                                           keep_prob: 1.0})
        test_writer.add_summary(summary_test, epoch)

        print('Loss: {:>2.6f}  Validation Accuracy: {:>2.3f}%'.format(cost_stats, valid_accuracy * 100))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, './saver/single_batch/' + version)


# Train on Full CIFAR-10 Batch #

# print('Training...')
#
# with tf.Session() as sess:
#     # Merge all the summaries and write them
#     merged = tf.summary.merge_all()
#     train_writer = tf.summary.FileWriter('./log/full_batch/' + version + '/train', sess.graph)
#     test_writer = tf.summary.FileWriter('./log/full_batch/' + version + '/test')
#
#     # Initializing the variables
#     sess.run(tf.global_variables_initializer())
#
#     # Training cycle
#     for epoch in range(epochs):
#         # Loop over all batches
#         n_batches = 5
#         for batch_i in range(1, n_batches + 1):
#             for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
#                 train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
#             print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#
#             summary_train, cost_stats = sess.run([merged, cost],
#                                                  feed_dict={x: batch_features,
#                                                             y: batch_labels,
#                                                             keep_prob: 1.0}, )
#             train_writer.add_summary(summary_train, epoch)
#
#             summary_test, valid_accuracy = sess.run([merged, accuracy],
#                                                     feed_dict={x: valid_features,
#                                                                y: valid_labels,
#                                                                keep_prob: 1.0})
#             test_writer.add_summary(summary_test, epoch)
#
#             print('Loss: {:>2.6f}  Validation Accuracy: {:>2.3f}%'.format(cost_stats, valid_accuracy * 100))
#
#     # Save Model
#     saver = tf.train.Saver()
#     save_path = saver.save(sess, './saver/full_batch/' + version)
