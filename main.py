from __future__ import division, print_function, absolute_import
from music21 import converter,instrument, duration, chord, stream # or import *
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from os import walk

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
from random import randint
import os
import time
from keras import backend as K

start = time.clock()
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
sec_train = mnist.train.next_batch(128)[1][0]
train_128 = []
print("------------------ Initializing\n\n\n\n\n")
# print(mnist.train.next_batch(128)[1][0])
# exit(0)

allnotesl = []
notes = []

midis = []
for (dirpath, dirnames, filenames) in walk("midis"):
    midis.extend(filenames)
    break

for midi in midis:
    if midi == "Someone_in_the_Crowd.mid":
        s = converter.parse('midis/{}'.format(midi))
        print("Preparing {}".format(midi))

        i = 0
        for el in s.recurse():
            dur = el.duration.quarterLength
            note = str(el).split(" ")[-1]
            note = note.replace(">", "")

            if dur > 10:
                el.duration.quarterLength = 5

            if dur <= 0.1:
                el.duration.quarterLength = 0.1

            if 'Instrument' in el.classes: # or 'Piano'
                el.activeSite.replace(el, instrument.Harp())

            notelength = [note, el.duration.quarterLength]
            allnotesl.append(notelength)

        # s.write('midi', 'example.mid')


        for notel in allnotesl:
            if isinstance(notel[0], str):
                notes.append(str(notel[0]))

t = Tokenizer(lower=False, filters="")
# fit the tokenizer on the documents
oof = t.fit_on_texts(notes)


# summarize what was learned
print("Keras Length, ", t.document_count)
print("Allnotes length: ", len(allnotesl))
out = ""
i = 0
# for note in allnotesl:
#     if i % 10 == 0:
#         out += "\n\n{}. {}".format(i, note[0])
#     else:
#         out += "  {}. {}".format(i, note[0])
#     i+=1
# print(out)
#
# with open("ados.txt", "w") as file:
#     file.writelines(out)

#print(t.word_index)
#print(t.index_word)

found_notes = 0
bit128_collection = []
for notel in allnotesl:
    for key, item in t.index_word.items():
        if item == notel[0]:
            bit128_collection.append([key, notel[1]])
            found_notes += 1
    if found_notes % 128 == 0:
        train_128.append(bit128_collection)
        bit128_collection = []
# if bit128_collection:
#     train_128.append(bit128_collection)

print("successfully gathered training data...\nFound {}/{} notes".format(found_notes, len(allnotesl)))
for i in range(len(train_128)):
    print("{}.Array length is {}".format(i, len(train_128[i])))

train_128 = np.array(train_128)
# k_128 is converted to be used as a machine learning variable
k_128 = K.variable(value=train_128, dtype='float64', name='k_128')

# MACHINE LEARNING PART -------------------
print("---------- Machine Learning\n\n\n\n\n")

num_steps = 1000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784  # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100  # Noise data points


# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}


# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Discriminator
def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Build Networks
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
             biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

now = datetime.datetime.now()

mod_dir = now.strftime("%Y-%m-%d--%H:%M")
img_dir = now.strftime("%Y-%m-%d--%H:%M")
# os.mkdir("trained/{}".format(img_dir))
# os.mkdir("model/{}".format(mod_dir))

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of midi data (only notes are needed, not labels)
        print(k_128.eval())
        exit(0)

        batch_x, _ = k_128.eval()
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

    # Generate images from noise, using the generator network.
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            print("DONE {}".format(j))
    #         img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
    #                          newshape=(28, 28, 3))
    #         saveimg = Image.fromarray(img, mode='RGB')
    #         saveimg.save("trained/{}/number{}.png".format(img_dir, randint(0, 1000000)))
    #
    # save_path = saver.save(sess, "model/{}/gan.ckpt".format(mod_dir))
end = time.clock()
print("Done in -- {0:.2f} seconds".format(end - start))
