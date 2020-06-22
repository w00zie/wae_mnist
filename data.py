import tensorflow as tf

def get_dataset(train_buffer=60000, test_buffer=10000, batch_size=100):

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).\
        shuffle(train_buffer).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).\
        shuffle(test_buffer).batch(batch_size)

    return (train_dataset, test_dataset)

if __name__ == "__main__":
    batch_size = 64
    train_set, test_set = get_dataset(batch_size=batch_size)
    print('Train')
    for batch in train_set:
        pass
    print('Test')
    for batch in test_set:
        pass
    