import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import Flatten, Reshape

def conv2D(h_dim=32, kernel_size=[7,7], use_bias=False,
           kernel_init="TruncatedNormal", strides=[2,2],
           padding="same", **kwargs):
    return Conv2D(filters=h_dim, kernel_size=kernel_size,
                         use_bias=use_bias, kernel_initializer=kernel_init,
                         strides=strides, padding=padding, **kwargs)

def conv2DT(h_dim=32, kernel_size=[7,7], use_bias=False,
            kernel_init="TruncatedNormal", strides=[2,2],
            padding="same", **kwargs):
    return Conv2DTranspose(filters=h_dim, kernel_size=kernel_size,
                           use_bias=use_bias, kernel_initializer=kernel_init,
                           strides=strides, padding=padding, **kwargs)

def dense(units=32, kernel_init="TruncatedNormal", use_bias=True, **kwargs):
    return Dense(units=units, kernel_initializer=kernel_init,
                 use_bias=use_bias, **kwargs)

def normalization(norm_method='batch', **kwargs):
    if norm_method not in ['batch', 'layer']:
        raise ValueError
    if norm_method == 'batch':
        return BatchNormalization(**kwargs)
    elif norm_method == 'layer':
        return LayerNormalization(**kwargs)

def activation(func='relu', **kwargs):
    if func not in ['relu', 'leaky']:
        raise ValueError
    if func == 'relu':
        return ReLU(**kwargs)
    elif func == 'leaky':
        return LeakyReLU(**kwargs)

def encoder(config,
            norm_method='batch',
            func='relu'):

    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    kernel_size = config["kernel_size"]

    inputs = Input(shape=(28, 28, 1))
    x = conv2D(h_dim=h_dim, 
               kernel_size=kernel_size,
               strides=[2,2]) (inputs)
    x = normalization(norm_method=norm_method,
                      name='norm_1') (x)
    x = activation(func=func, name='act_1') (x)
    x = conv2D(h_dim=2*h_dim, 
               kernel_size=kernel_size,
               strides=[2,2]) (x)
    x = normalization(norm_method=norm_method,
                      name='norm_2') (x)
    x = activation(func=func, name='act_2') (x)
    x = conv2D(h_dim=4*h_dim, 
               kernel_size=kernel_size,
               strides=[2,2]) (x)
    x = normalization(norm_method=norm_method,
                      name='norm_3') (x)
    x = activation(func=func, name='act_3') (x)
    x = conv2D(h_dim=8*h_dim, 
               kernel_size=kernel_size,
               strides=[2,2]) (x)
    x = normalization(norm_method=norm_method,
                      name='norm_4') (x)
    x = activation(func=func, name='act_4') (x)
    x = Flatten() (x)
    outputs = dense(units=z_dim) (x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Encoder")

def decoder(config,
            norm_method='batch',
            func='relu'):

    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    kernel_size = config["conv_kernel_size"]

    inputs = Input(shape=(z_dim))
    x = dense(units=7*7*(8*h_dim)) (inputs)
    x = normalization(norm_method=norm_method) (x)
    x = activation(func=func) (x)
    x = Reshape((7, 7, 8*h_dim)) (x)
    x = conv2DT(h_dim=4*h_dim, kernel_size=kernel_size, strides=[2,2]) (x)
    x = normalization(norm_method=norm_method) (x)
    x = activation(func=func, name='act_1') (x)
    x = conv2DT(h_dim=2*h_dim, kernel_size=kernel_size, strides=[2,2]) (x)
    x = normalization(norm_method=norm_method) (x)
    x = activation(func=func, name='act_2') (x)
    outputs = conv2DT(h_dim=1, kernel_size=kernel_size, strides=[1,1],
                activation="sigmoid") (x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Decoder")

if __name__ == "__main__":
    h_dim, z_dim = 32, 16
    encoder = encoder(h_dim=h_dim, z_dim=z_dim)
    decoder = decoder(h_dim=h_dim, z_dim=z_dim)
    
    inputs = tf.random.uniform(shape=[32, 28, 28, 1])
    pz = tf.random.uniform(shape=[32, z_dim])
    
    qz = encoder(inputs)
    x_hat = decoder(qz)
    
    assert x_hat.shape == inputs.shape, 'Shape error'

    print(encoder.summary())
    print(decoder.summary())
