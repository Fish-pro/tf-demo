import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version:", tf.__version__)
print("Eager mode:",tf.executing_eagerly())
print("Hub Version:", hub.__version__)
print("Tfds Version:", tfds.__version__)
print("GPU is ", "available" if tf.config.list_physical_devices('GPU') else "not available")

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train', 'unsupervised', 'test'),
    as_supervised=True
)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch)
# print(train_labels_batch)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# print(hub_layer(train_examples_batch[:3]))

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer="adam",
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name,value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

