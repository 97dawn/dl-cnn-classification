'''
@author: DoeunKim
@reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
@attention: You can run this program locally. 
            You must not quit this program until it terminates by itself.
            file structure:
                Q1.py
                man
                woman
                test
                    -man
                    -woman
                training
                    -man
                    -woman
'''
import os
import shutil
import random
import tensorflow as tf


# Image Parameters
N_CLASSES = 2 
IMG_HEIGHT = 32 
IMG_WIDTH = 32 
CHANNELS = 3 
NUM_IMGS = 705


# Parameters
LEARNING_RATE = 0.001
NUM_STEPS = 1500
BATCH_S = 141
DISPLAY_STEP = 100


def seperate_dataset():
    # Create empty training and test directories
    if not os.path.exists("training/man"):
        os.makedirs("training/man")
    if not os.path.exists("training/woman"):
        os.makedirs("training/woman")
    if not os.path.exists("test/man"):
        os.makedirs("test/man")
    if not os.path.exists("test/woman"):
        os.makedirs("test/woman")
    
    # Copy 80% of shuffled man and woman images to the training dir
    m_imgs = [f for f in os.listdir("man/")]
    w_imgs = [f for f in os.listdir("woman/")]
    random.shuffle(m_imgs)
    random.shuffle(w_imgs)
    for file in m_imgs[0:int(NUM_IMGS * 0.8)]:
        shutil.move('man/' + file, "training/man/" + file)
    for file in w_imgs[0:int(NUM_IMGS * 0.8)]:
        shutil.move('woman/' + file, "training/woman/" + file)
        
    # Move rest of man and woman images to the test dir
    for file in m_imgs[int(NUM_IMGS * 0.8):]:
        shutil.move('man/' + file,'test/man/' + file)
    for file in w_imgs[int(NUM_IMGS * 0.8):]:
        shutil.move('woman/' + file, 'test/woman/' + file)
        
def return_dataset():
    # Remove copied files
    for f in os.listdir("training/man"):
        shutil.move('training/man/'+f, 'man/' + f)
    for f in os.listdir("training/woman"):
        shutil.move('training/woman/'+f, 'woman/'+f)
    for f in os.listdir('test/man/'):
        shutil.move('test/man/'+f, 'man/'+f)
    for f in os.listdir('test/woman/'):
        shutil.move('test/woman/'+f,'woman/'+f)
        
    # Remove empty training directories
    if os.path.exists("training"):
        os.rmdir("training/man")
        os.rmdir("training/woman")
        os.rmdir("training")
    if os.path.exists("test"):
        os.rmdir("test/man")
        os.rmdir("test/woman")
        os.rmdir("test")
        
def read_images(man_dir, woman_dir, batch, mul):
    images, labels = list(), list()
    for f in os.listdir(man_dir):
        images.append(os.path.join(man_dir,f ))
        labels.append(0)
    for f in os.listdir(woman_dir):
        images.append(os.path.join(woman_dir,f ))
        labels.append(1)
     
    # Convert to Tensor
    images = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([images, labels], shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_png(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image /(32+32-0.5) - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch,
                          capacity=batch * mul,
                          num_threads=4)

    return X, Y

# Create model

def cnn(x, reuse, is_training):

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet',reuse=reuse):

        # Convolution Layer #1
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[2,2],
                                 padding="SAME", activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], 
                                        padding="SAME", strides=1)
        
        dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.7, training=is_training)
        # Convolution Layer #2
        conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], 
                                 padding="SAME", strides = 2, activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],
                                        padding="SAME", strides= 2)
        dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=is_training)
        
        # Fully connected layer 
        fc1 = tf.contrib.layers.flatten(dropout2)
        fc1 = tf.layers.dense(fc1, 512)
        fc1 = tf.layers.dropout(fc1, rate=0.75, training=is_training)
        

        # Output layer, class prediction
        out = tf.layers.dense(fc1, N_CLASSES)
        out = tf.nn.softmax(out) if not is_training else out

    return out

def main():
    # Separate training data set and test data set
    seperate_dataset()
    X_train, Y_train = read_images("training/man/","training/woman/", 141, 8)
    X_test, Y_test = read_images("test/man/", "test/woman/",47 , 6 )
    
    # Create a graph for training
    logits_train = cnn(X_train, False, True)
    
    # Define loss and optimizer (with train logits, for dropout to take effect)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y_train))
    train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Create another graph for testing that reuse the same weights
    logits_test = cnn(X_test, True, False)
    
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_test,tf.int64 ))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
    
        # Start the data queue
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess, coord=coord)
    
        # Training cycle
        for step in range(1, NUM_STEPS+1):
            sess.run(train)    
            if step % DISPLAY_STEP == 0:
                _, loss, acc = sess.run([train, cost, accuracy])
                print("Step " + str(step) + ", Loss= " + \
                      "{:.4f}".format(loss) + ", Accuracy= " + \
                      "{:.3f}".format(acc))
        coord.request_stop()
        coord.join(threads)
    
    return_dataset()
    
if __name__ == '__main__':
    main()