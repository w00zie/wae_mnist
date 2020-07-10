import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, 
                 h_dim=32, 
                 z_dim=20,
                 kernel_size=(6,6),
                 kernel_init = "TruncatedNormal",
                 name="encoder",
                 **kwargs):

        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder = tf.keras.Sequential(
            [   
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=h_dim, 
                                       kernel_size=kernel_size,
                                       use_bias=False,
                                       strides=(2,2),
                                       kernel_initializer=kernel_init,
                                       padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=2*h_dim, 
                                       kernel_size=kernel_size,
                                       use_bias=False,
                                       strides=(2,2),
                                       kernel_initializer=kernel_init,
                                       padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=4*h_dim, 
                                       kernel_size=kernel_size,
                                       use_bias=False,
                                       strides=(2,2),
                                       kernel_initializer=kernel_init,
                                       padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=8*h_dim, 
                                       kernel_size=kernel_size,
                                       use_bias=False,
                                       strides=(2,2),
                                       kernel_initializer=kernel_init,
                                       padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=z_dim, 
                                      kernel_initializer=kernel_init)
            ], name="encoder")
        
        tv = tf.reduce_sum([tf.reduce_prod(v.shape) for 
                            v in self.encoder.trainable_variables])
        tf.print(f"Encoder has {tv} trainable params.")

    #@tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.encoder(inputs)


class Decoder(tf.keras.Model):
    def __init__(self, 
                 h_dim=32, 
                 z_dim=20,
                 kernel_size=(6,6),
                 kernel_init = "TruncatedNormal",
                 name="decoder",
                 **kwargs):

        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(z_dim,)),
                tf.keras.layers.Dense(units=7*7*(8*h_dim), 
                                      kernel_initializer=kernel_init),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape((7, 7, 8*h_dim)),
                tf.keras.layers.Conv2DTranspose(filters=4*h_dim, 
                                                kernel_size=kernel_size,
                                                use_bias=False,
                                                strides=(2,2),
                                                padding="same",
                                                kernel_initializer=kernel_init),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(filters=2*h_dim, 
                                                kernel_size=kernel_size,
                                                use_bias=False,
                                                strides=(2,2),
                                                padding="same",
                                                kernel_initializer=kernel_init),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(filters=1, 
                                       kernel_size=kernel_size,
                                       strides=(1,1),
                                       padding="same",
                                       kernel_initializer=kernel_init,
                                       activation="sigmoid"),
            ], name="decoder")

        tv = tf.reduce_sum([tf.reduce_prod(v.shape) for 
                            v in self.decoder.trainable_variables])
        tf.print(f"Decoder has {tv} trainable params.")

    #@tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.decoder(inputs)


if __name__ == "__main__":

    batch = tf.random.uniform(shape = [32, 28, 28, 1])

    h_dim, z_dim, units = 64, 16, 512

    kernel_size = (5, 5)
    kernel_init = "TruncatedNormal"
    #kernel_init = "he_normal"
    #kernel_init = "glorot_normal"
    encoder = Encoder(h_dim=h_dim, z_dim=z_dim, kernel_size=kernel_size, kernel_init=kernel_init)
    decoder = Decoder(h_dim=h_dim, z_dim=z_dim, kernel_size=kernel_size, kernel_init=kernel_init)

    z = tf.random.normal(shape=(1, z_dim))
    z_hat = encoder(batch)
    x_hat = decoder(z_hat)
    
    encoder.encoder.summary()
    decoder.decoder.summary()
    
    tf.print(f"{batch.shape} -> {z_hat.shape} -> {x_hat.shape}")

    tf.print("\n"+30*"-"+"AUTOENCODER"+30*"-"+"\n")

    mse_loss = tf.keras.losses.MSE(y_pred=tf.reshape(x_hat, (-1, 28*28)), 
                               y_true=tf.reshape(batch, (-1, 784)))
    mse_loss_2 = tf.keras.losses.MSE(y_true=batch, y_pred=x_hat)
    mse_loss_3 = tf.math.reduce_sum(tf.math.square(batch - x_hat), axis=[1,2,3])
    mse_loss_4 = tf.reduce_mean(tf.keras.losses.MSE(y_true=batch, y_pred=x_hat), axis=[1,2])

    tf.print(f"From shape {mse_loss.shape} -> mean -> MSE = {tf.reduce_mean(mse_loss)}")
    tf.print(f"From shape {mse_loss_2.shape} -> mean -> MSE = {tf.reduce_mean(mse_loss_2)}")
    tf.print(f"From shape {mse_loss_3.shape} -> mean -> MSE = {tf.reduce_mean(mse_loss_3)}")
    tf.print(f"From shape {mse_loss_4.shape} -> mean -> MSE = {tf.reduce_mean(mse_loss_4)}")