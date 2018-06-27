import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# get the data
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
plt.rcdefaults()

# Parameterized mnist learning. See effectiveness of different activation functions.
# Loosely based on this tutorial: http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
def learn_to_recognize_mnist(epochs = 10, learning_rate = 0.01, batch_size = 100, activator = tf.nn.relu, activator_name = '?',weights = tf.Session().run(tf.random_normal([784,300],stddev=0.03))):
    tf.reset_default_graph()
    tf.set_random_seed(55) # does not work
    writer = tf.summary.FileWriter('./summary', graph = tf.get_default_graph())
    
    with tf.name_scope('input'):
        mnist_pixels = tf.placeholder(tf.float32, [None, 784], name = 'mnist_pixels')  # 28 * 28 = 784 inputs

    with tf.name_scope('output'):
        result_one_hot = tf.placeholder(tf.float32, [None, 10], name = 'result_one_hot')   # 10 possible results with one-hot

    with tf.name_scope('hidden_layer_input'):
        #hidden_layer_input_weights = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='hidden_layer_input_weights')
        hidden_layer_input_weights = tf.Variable(weights , name = 'hidden_layer_input_weights')
        pixels_times_weights = tf.matmul(mnist_pixels, hidden_layer_input_weights, name = 'pixels_times_weights')
        hidden_layer_input_bias = tf.Variable(tf.random_normal([300]), name='hidden_layer_input_bias')
        hidden_layer_value = tf.add(pixels_times_weights, hidden_layer_input_bias, name = 'hidden_layer_value') # pixels times weights plus bias

        with tf.name_scope('hidden_layer_input_activation'):
            hidden_layer_input_activated = activator(hidden_layer_value)
            #print('hidden_layer_input_activated.shape',hidden_layer_input_activated.shape)
            #print('hidden_layer_input_activated',hidden_layer_input_activated)

    with tf.name_scope('hidden_layer_output'):
        hidden_layer_output_weights = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name = 'hidden_layer_output_weights')
        hidden_layer_times_weights = tf.matmul(hidden_layer_input_activated, hidden_layer_output_weights)
        hidden_layer_output_bias = tf.Variable(tf.random_normal([10]), name='hidden_layer_output_bias')
        hidden_layer_output_value = tf.add(hidden_layer_times_weights, hidden_layer_output_bias)

        with tf.name_scope('hidden_layer_output_activation'):
            hidden_layer_output = tf.nn.softmax(hidden_layer_output_value)

    with tf.name_scope('training'):
        hidden_layer_output_clipped = tf.clip_by_value(hidden_layer_output, 1e-10, 0.9999999) # otherwise nan results
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(
            result_one_hot * tf.log(hidden_layer_output_clipped) + (1 - result_one_hot) * tf.log(1 - hidden_layer_output_clipped)
            , axis=1))
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.name_scope(activator_name + '_accuracy'):
        distance = tf.equal(tf.argmax(result_one_hot, 1), tf.argmax(hidden_layer_output, 1))
        accuracy = tf.reduce_mean(tf.cast(distance, tf.float32))
        tf.summary.scalar('run', accuracy)  #summary
        merged = tf.summary.merge_all()

    # run
    session = tf.Session()
    with session:
        session.run(tf.global_variables_initializer())
        number_of_batches = int(len(mnist.train.labels) / batch_size)
        
        for epoch in range(epochs):
            average_cost = 0
            for a_batch in range(number_of_batches):
                mnist_input_pixels, mnist_correct_labels = mnist.train.next_batch(batch_size=batch_size)
                training_inputs = { mnist_pixels: mnist_input_pixels, result_one_hot: mnist_correct_labels }
                _, cost = session.run([optimiser, cross_entropy], feed_dict = training_inputs)
                average_cost += cost / number_of_batches

            # measure epoch result
            accuracy_inputs = { mnist_pixels: mnist.test.images, result_one_hot: mnist.test.labels }
            summary = session.run(merged, feed_dict = accuracy_inputs)
            writer.add_summary(summary, epoch)
            print("Epoch ", (epoch), " has distance", average_cost)

        writer.add_graph(session.graph)
        accuracy_inputs = { mnist_pixels: mnist.test.images, result_one_hot: mnist.test.labels }
        result = session.run(accuracy, feed_dict=accuracy_inputs)
        print('Trained to accuracy ', result)

   #     session.close()
   #     writer.close()
   
   #     return result

    session.close()
    writer.close()
    return result

# See description of ramp here http://iopscience.iop.org/article/10.1088/1757-899X/224/1/012054/pdf
# It's a simple clip of values outside 0..1
def ramp(x):
    return tf.clip_by_value(x, clip_value_min=0.001, clip_value_max=0.999, name='ramp')

def swish(x):
    with tf.name_scope('swish'):
        swished = tf.nn.sigmoid(x) * x
        return swished

activators = {
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'softmax': tf.nn.softmax,
    'tanh': tf.nn.tanh,
    'ramp': ramp,
    'swish': swish,
    'sigmoid': tf.nn.sigmoid,
    'leaky_relu':tf.nn.leaky_relu
}

activator_list = []
performance = []
o_weights = tf.random_normal([784, 300], stddev=0.03)
original_weights = tf.Session().run(o_weights)

print('\n\n---------------------------------')
print('-----   test activation functions -----')

for activator_name in activators:
    activator = activators[activator_name]
    print('\n---', activator_name, '---')
    use_weights = original_weights
    result = learn_to_recognize_mnist(epochs = 7, activator_name = activator_name, activator = activator,weights=use_weights)
    activator_list.append(activator_name)
    performance.append(result)

print('\n---------------------------------\n\n')

# Adding a bar graph to the mix

y_pos = np.arange(len(activator_list))
plt.bar( y_pos , performance , align='center' , alpha = 0.5 )
plt.xticks( y_pos , activator_list )
plt.ylabel('Accuracy')
plt.title('Chart Comparison')

plt.show()

