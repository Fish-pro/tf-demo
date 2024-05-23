import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 只取前1000个样本
train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]

# 数据预处理：将图像展平并归一化
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu',input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


# model = create_model()
#
# model.summary()
#
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# # Train the model with the new callback
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])  # Pass callback to training
#
# # This may generate warnings related to saving the state of the optimizer.
# # These warnings (and similar warnings throughout this notebook)
# # are in place to discourage outdated usage, and can be ignored.
# print(checkpoint_dir)
# print(os.listdir(checkpoint_dir))
#
# model.load_weights(checkpoint_path)
#
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#
# batch_size = 32
#
# # Calculate the number of batches per epoch
# import math
# n_batches = len(train_images) / batch_size
# n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer
#
# # Create a callback that saves the model's weights every 5 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     save_freq=5*n_batches)
#
# # Create a new model instance
# model = create_model()
#
# # Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))
#
# # Train the model with the new callback
# model.fit(train_images,
#           train_labels,
#           epochs=50,
#           batch_size=batch_size,
#           callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)
#
# print(os.listdir(checkpoint_dir))
#
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# print(latest)
#
# model = create_model()
# model.load_weights(latest)
#
# loss, acc = model.evaluate(test_images,test_labels,verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# # Save the weights
# model.save_weights('./checkpoints/my_checkpoint')
#
# # Create a new model instance
# model = create_model()
#
# # Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')
#
# # Evaluate the model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
#
# model.save("my_model.keras")
#
# new_model = tf.keras.models.load_model("my_model.keras")
# new_model.summary()
#
# # Evaluate the restored model
# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
#
# print(new_model.predict(test_images).shape)


# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
# model.save('saved_model/my_model')
#
# new_model = tf.keras.models.load_model('saved_model/my_model')
# new_model.summary()
#
# # Evaluate the restored model
# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
#
# print(new_model.predict(test_images).shape)


model = create_model()
model.fit(train_images, train_labels, epochs=5)

model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))