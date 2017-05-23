import tensorflow as tf

def weight_variable(name, shape):
    """weight_variable generates a weight variable of a given shape."""
    return tf.get_variable(name, shape=shape,
    initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def inference_color(x):

    W_conv1 = weight_variable("conv1", [5, 5, 3, 64])
    H_conv1 = tf.nn.relu(conv2d(x, W_conv1))

    W_conv2 = weight_variable("conv2", [5, 5, 64, 64])
    H_conv2 = tf.nn.relu(conv2d(H_conv1, W_conv2))

    W_conv3 = weight_variable("conv3", [1, 1, 64, 48])
    H_conv3 = tf.nn.relu(conv2d(H_conv2, W_conv3))

    W_conv4 = weight_variable("conv4", [5, 5, 48, 32])
    H_conv4 = tf.nn.relu(conv2d(H_conv3, W_conv4))

    W_conv5 = weight_variable("conv5", [5, 5, 32, 32])
    H_conv5 = tf.nn.relu(conv2d(H_conv4, W_conv5))

    W_conv6 = weight_variable("conv6", [5, 5, 32, 3])
    H_conv6 = tf.nn.relu(conv2d(H_conv5, W_conv6))

    return H_conv6

def inference_gray(x):

    W_conv1 = weight_variable("conv1", [5, 5, 1, 64])
    H_conv1 = tf.nn.relu(conv2d(x, W_conv1))

    W_conv2 = weight_variable("conv2", [5, 5, 64, 64])
    H_conv2 = tf.nn.relu(conv2d(H_conv1, W_conv2))

    W_conv3 = weight_variable("conv3", [1, 1, 64, 48])
    H_conv3 = tf.nn.relu(conv2d(H_conv2, W_conv3))

    W_conv4 = weight_variable("conv4", [5, 5, 48, 32])
    H_conv4 = tf.nn.relu(conv2d(H_conv3, W_conv4))

    W_conv5 = weight_variable("conv5", [5, 5, 32, 32])
    H_conv5 = tf.nn.relu(conv2d(H_conv4, W_conv5))

    W_conv6 = weight_variable("conv6", [5, 5, 32, 1])
    H_conv6 = tf.nn.relu(conv2d(H_conv5, W_conv6))

    return H_conv6

def loss(logits, img):
    return tf.nn.l2_loss(tf.subtract(logits, img))

def training(loss, learning_rate, global_step):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
