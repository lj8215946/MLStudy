import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# Neural Network Input Layer & Output Layer
CL_ML_INPUT_NODE_COUNT = 784
CL_ML_OUTPUT_NODE_COUNT = 10

# Define Hidden Layer Node Count
CL_ML_HIDDEN_LAYER_NODE_COUNT = 1000

# Batch Size
CL_ML_LEARNING_BATCH_SIZE = 100

# Learning Rate
CL_ML_LEARNING_RATE_INIT_VALUE = 0.8
CL_ML_LEARNING_RATE_INIT_DECAY_FACTOR = 0.99

# Regularization Factor
CL_ML_LEARNING_REGULARIZATION_FACTOR = 0.0001

# Training Steps
CL_ML_TRAINING_STEPS = 30000

# All Trainable Factor Decay Factor
CL_ML_MOVING_AVERAGE_DECAY_FACTOR = 0.99


# Input Value & Weights & Biases,Output Whole Neural Network
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):

    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# Train Method
def train(mnist):
    x = tf.placeholder(tf.float32, [None, CL_ML_INPUT_NODE_COUNT], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, CL_ML_OUTPUT_NODE_COUNT], name='y-input')

    # generate hidden layer parameters
    weights1 = tf.Variable(tf.truncated_normal([CL_ML_INPUT_NODE_COUNT, CL_ML_HIDDEN_LAYER_NODE_COUNT], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[CL_ML_HIDDEN_LAYER_NODE_COUNT]))

    # generate output layer parameters
    weights2 = tf.Variable(tf.truncated_normal([CL_ML_HIDDEN_LAYER_NODE_COUNT, CL_ML_OUTPUT_NODE_COUNT], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[CL_ML_OUTPUT_NODE_COUNT]))

    # compute result without decay
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # define train step
    global_train_step = tf.Variable(0, trainable=False)

    # define train parameter decay
    variable_averages = tf.train.ExponentialMovingAverage(CL_ML_MOVING_AVERAGE_DECAY_FACTOR, global_train_step)

    # define train parameter decay op
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # compute result with decay
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # compute total cost function
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)

    # compute mean of cost function
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # define regularizer using L2
    regularizer = tf.contrib.layers.l2_regularizer(CL_ML_LEARNING_REGULARIZATION_FACTOR)

    # compute regularization
    regularization = regularizer(weights1) + regularizer(weights2)

    # cost function with regularization
    loss = cross_entropy_mean + regularization

    # define learning rate
    learning_rate = tf.train.exponential_decay(CL_ML_LEARNING_RATE_INIT_VALUE,
                                               global_train_step,
                                               mnist.train.num_examples / CL_ML_LEARNING_BATCH_SIZE,
                                               CL_ML_LEARNING_RATE_INIT_DECAY_FACTOR)

    # define train step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_train_step)

    # define train op
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # compute correct vector
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # accuracy using parameter decay
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # prepare validate feed
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # test validate feed
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        # iteration training
        for step_index in range(CL_ML_TRAINING_STEPS):

            # print when step index % 1000 = 0
            if step_index % 1000 == 0:

                # compute validate accuracy
                validate_acc = sess.run(accuracy, feed_dict=validate_feed,)
                print("After %d training steps, validation accuracy using average model is %g"
                      % (step_index, validate_acc))

            # generate new batch of training data and train again
            xs, ys = mnist.train.next_batch(CL_ML_LEARNING_BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # after train, test accuracy on test data
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" % (CL_ML_TRAINING_STEPS, test_acc))


# main entry
def main(argv=None):
    print("Begin main")

    # define mnist and download data
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

# tensor flow main
if __name__ == '__main__':
    tf.app.run()
