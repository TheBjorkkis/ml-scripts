from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Download the data.
mnist_data = input_data.read_data_sets( 
		'mnist_data',
		one_hot=True )

# Build up the perceptron.
input_size = 784
no_classes = 10
batch_size = 100
total_batches = 200

# Defining placeholders.
x_input = tf.placeholder( 
		tf.float32,
		shape = [
				None,
				input_size ] )
y_input = tf.placeholder(
		tf.float32,
		shape = [
				None,
				no_classes ] )

# Defining variables for a fully connected layer.
weights = tf.Variable( 
		tf.random_normal( [ 
				input_size, 
				no_classes ] ) )
bias = tf.Variable( tf.random_normal( [ no_classes ] ) )
logits = tf.matmul( 
		x_input, 
		weights ) + bias

# Train a optimiser.
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( 
		labels = y_input,
		logits = logits )
loss_operation = tf.reduce_mean( softmax_cross_entropy )
optimiser = tf.train.GradientDescentOptimizer( 
		learning_rate = 0.5 ).minimize( loss_operation )

# Build a session for training.
session = tf.Session()
session.run( tf.global_variables_initializer() )

# Read the data in batches and train the model.
# Weights are being updated.
for batch_no in range(total_batches):
	mnist_batch = mnist_data.train.next_batch(batch_size)
	_, loss_value = session.run( [ 
		optimiser, loss_operation ],
		feed_dict = {
				x_input: mnist_batch[ 0 ],
				y_input: mnist_batch[ 1 ] } )

# Define placeholders for predictions before test data is run.
predictions = tf.argmax(
		logits,
		1 )
correct_predictions = tf.equal( 
		predictions,
		tf.argmax( 
				y_input, 
				1 ) )
accuracy_operation = tf.reduce_mean( 
		tf.cast(
				correct_predictions, 
				tf.float32 ) )

# Use the testing data to calculate the accuracy.
test_images, test_labels = mnist_data.test.images, mnist_data.test.labels
accuracy_value = session.run( 
		accuracy_operation, 
		feed_dict = { 
				x_input: test_images,
				y_input: test_labels } )
print('Accuracy: ', accuracy_value )
session.close()

