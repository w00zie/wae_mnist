import tensorflow as tf
from model import Encoder, Decoder, Discriminator
from data import get_dataset
import os
import json
from datetime import datetime, timedelta
from time import time


class Train:

    def __init__(self, 
                 batch_size=100, 
                 epochs=10,
                 h_dim=32,
                 conv_kernel_size=(6,6),
                 kernel_init="glorot_normal",
                 disc_units=256,
                 z_dim=10,
                 sigma_z=1.,
                 disc_lr=1e-4,
                 enc_dec_lr=1e-5,
                 lmbda=0.8):

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.sigma_z = sigma_z
        self.lmbda = lmbda
        
        # Experiment directory
        self.logdir = "runs/" + datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
        config = {"h_dim": h_dim,
                  "z_dim": z_dim,
                  "epochs": epochs,
                  "conv_kernel_size": conv_kernel_size,
                  "disc_units": disc_units,
                  "sigma_z": sigma_z,
                  "discriminator learning rate": disc_lr,
                  "encoder/decoder learning rate": enc_dec_lr,
                  "exp_dir": self.logdir,
                  "lambda": lmbda}

        self.writer = tf.summary.create_file_writer(self.logdir)
        with self.writer.as_default():
            text = """# Hyperparameters used
                      1. h_dim = {}
                      2. z_dim = {}
                      3. kernel_size = {}
                      4. discriminator_units = {}
                      5. sigma_z = {}
                      6. d_lr = {}
                      7. ae_lr = {}
                      8. lambda = {}""".format(h_dim, z_dim, 
                                                conv_kernel_size,
                                                disc_units, sigma_z, disc_lr,
                                                enc_dec_lr, lmbda)
            tf.summary.text("Hyperparams", text, step=0)
        self.writer.flush()
        os.mkdir(self.logdir+"/img")
        with open(self.logdir+"/config.json", "w") as f:
            json.dump(config, f)

        # Models --------------------------------------------------------------
        self.encoder = Encoder(h_dim=h_dim, 
                               z_dim=z_dim, 
                               kernel_size=conv_kernel_size, 
                               kernel_init=kernel_init)
        self.decoder = Decoder(h_dim=h_dim, 
                               z_dim=z_dim, 
                               kernel_size=conv_kernel_size,
                               kernel_init=kernel_init)
        self.discriminator = Discriminator(units=disc_units, 
                                           z_dim=z_dim, kernel_init=kernel_init)

        # Optimizers ----------------------------------------------------------
        ae_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=enc_dec_lr,
            decay_steps=10000,
            decay_rate=0.95)
        disc_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=disc_lr,
            decay_steps=10000,
            decay_rate=0.95)

        self.enc_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.dec_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.disc_optim = tf.keras.optimizers.Adam(disc_scheduler)

        # Data -----------------------------------------------------------------
        tf.print("Loading data...")
        self.train_dataset, self.test_dataset = get_dataset(batch_size=batch_size)
        tf.print("Done.")

        # Metric trackers ------------------------------------------------------
        self.avg_d_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_d_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_d_z_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_mse_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)

    @tf.function
    def alternative_disc_loss(self, batch: tf.Tensor) -> tf.Tensor:
        z = tf.random.normal(shape=(tf.shape(batch)[0], self.z_dim),
                             mean=0., 
                             stddev=tf.sqrt(self.sigma_z))
        d_z = self.discriminator(z, training=True)
        d_z_hat = self.discriminator(self.encoder(batch, training=True), training=True)
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_z), 
                                                                       y_pred=d_z))
        fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(d_z_hat), 
                                                                       y_pred=d_z_hat))
        return self.lmbda*(real_loss + fake_loss)


    @tf.function
    def alternative_ae_loss(self, batch: tf.Tensor) -> tf.Tensor:
        z_hat = self.encoder(batch, training=True)
        x_hat = self.decoder(z_hat, training=True)
        d_z_hat = self.discriminator(z_hat, training=True)
        c = tf.math.reduce_mean(tf.reduce_sum(tf.square(batch - x_hat), axis=[1,2,3]))
        penalty = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_z_hat), 
                                                                     y_pred=d_z_hat))
        return c + self.lmbda*penalty

    @tf.function
    def train_discriminator(self, batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            disc_loss = self.alternative_disc_loss(batch)
        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optim.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def train_enc_dec(self, batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            enc_dec_loss = self.alternative_ae_loss(batch)
        enc_grads = enc_tape.gradient(enc_dec_loss, self.encoder.trainable_variables)
        dec_grads = dec_tape.gradient(enc_dec_loss, self.decoder.trainable_variables)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))
        return enc_dec_loss

    def log_hist(self, step):
        for w in self.encoder.trainable_weights:
            tf.summary.histogram("Encoder/{}".format(w.name), w, step=step)
        for w in self.decoder.trainable_weights:
            tf.summary.histogram("Decoder/{}".format(w.name), w, step=step)
        for w in self.discriminator.trainable_weights:
            tf.summary.histogram("Discriminator/{}".format(w.name), w, step=step)

    def display_random_bar(self, epoch):
        z = tf.random.normal(shape=(8, self.z_dim),
                              mean=0., 
                              stddev=tf.sqrt(self.sigma_z))
        x_hat = self.decoder(z, training=False)
        tf.summary.image("Random", x_hat, step=epoch)
        img = tf.squeeze(tf.concat([x_hat[i,:,:,:] for i in range(8)], axis=0), axis=-1)
        plt.imsave(self.logdir+"/img/random_{}".format(epoch), img)
    
    def display_reconstruction(self, epoch):
        for batch in self.test_dataset.take(1):
            x_hat = self.decoder(self.encoder(batch, training=False), training=False)
            tf.summary.image("Reconstructed", x_hat, step=epoch)
            tf.summary.image("Real", batch, step=epoch)
            reals = tf.squeeze(tf.concat([batch[i,:,:,:] for i in range(8)], axis=0), axis=-1)
            fakes = tf.squeeze(tf.concat([x_hat[i,:,:,:] for i in range(8)], axis=0), axis=-1)
            img = tf.concat([reals, fakes], axis=1)
            plt.imsave(self.logdir+"/img/recon_{}".format(epoch), img)

    @tf.function
    def calc_discriminator_latent(self):
        self.avg_d_z_loss.reset_states()
        for _ in tf.range(100):
            z = tf.random.normal(shape=(self.batch_size, self.z_dim),
                                 mean=0.,
                                 stddev=tf.sqrt(self.sigma_z))
            d_z = self.discriminator(z, training=False)
            self.avg_d_z_loss(d_z)
        return self.avg_d_z_loss.result()

    @tf.function
    def validation_step(self):
        # Reset metrics
        self.avg_enc_dec_test_loss.reset_states()
        self.avg_mse_test_loss.reset_states()
        self.avg_d_test_loss.reset_states()
        # Run through an epoch of validation
        for batch in self.test_dataset:
            z_hat = self.encoder(batch, training=False)
            x_hat = self.decoder(z_hat, training=False)
            d_z_hat = self.discriminator(z_hat, training=False)
            c = tf.reduce_mean(tf.keras.losses.MSE(y_true=batch, y_pred=x_hat), axis=[1,2])
            penalty = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_z_hat), 
                                                          y_pred=d_z_hat)
            # Log metrics for every batch
            self.avg_d_test_loss(tf.reduce_mean(d_z_hat))
            self.avg_mse_test_loss(tf.reduce_mean(c))
            self.avg_enc_dec_test_loss(tf.reduce_mean(c + self.lmbda*penalty))
        return (
            self.avg_d_test_loss.result(), 
            self.avg_enc_dec_test_loss.result(), 
            self.avg_mse_test_loss.result()
        )

    @tf.function
    def train_step(self):
        # Reset metrics
        self.avg_d_train_loss.reset_states()
        self.avg_enc_dec_train_loss.reset_states()
        # Run through an epoch of training
        for batch in self.train_dataset:
            enc_dec_loss = self.train_enc_dec(batch)
            disc_loss = self.train_discriminator(batch)
            # Log metrics for every batch
            self.avg_d_train_loss(disc_loss)
            self.avg_enc_dec_train_loss(enc_dec_loss)
        return (
            self.avg_d_train_loss.result(), 
            self.avg_enc_dec_train_loss.result()
        )

    #@tf.function
    def train(self):
        with self.writer.as_default():
            #with tf.summary.record_if(self.log_train):
            start = time()
            for epoch in range(self.epochs):
                # Train for one epoch
                d_loss, e_d_loss = self.train_step()
                # Display a few bars
                if epoch % 2 == 0:
                    self.display_random_bar(epoch)
                    self.display_reconstruction(epoch)
                # Validate on the test set
                d_g_z, test_loss, mse = self.validation_step()
                # Calculate D(z)
                d_z = self.calc_discriminator_latent()
                # Logging
                tf.summary.scalar("Train/Discriminator Loss", 
                                  d_loss, step=epoch)
                tf.summary.scalar("Train/Encoder-Decoder Loss", 
                                  e_d_loss, step=epoch)
                tf.summary.scalar("Learning Rates/Encoder", 
                                  self.enc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Decoder", 
                                  self.dec_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Discriminator", 
                                  self.disc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Test/D(G(x))", d_g_z, step=epoch)
                tf.summary.scalar("Test/D(z)", d_z, step=epoch)
                tf.summary.scalar("Test/Encoder-Decoder Loss", 
                                  test_loss, step=epoch)
                tf.summary.scalar("Test/MSE", mse, step=epoch)

                self.log_hist(epoch)
                
                tf.print(f"Epoch {epoch}/{self.epochs-1}  -  [d_loss = {d_loss.numpy()}]  -  [ae_loss = {e_d_loss.numpy()}]  -  {datetime.now().strftime('%H:%M:%S')}")
       
        print(f"Training took {timedelta(seconds=time()-start)}")
        
        os.mkdir(self.logdir+"/models")
        os.mkdir(self.logdir+"/models/encoder")
        os.mkdir(self.logdir+"/models/decoder")
        os.mkdir(self.logdir+"/models/discriminator")
        self.encoder.save_weights(self.logdir+"/models/encoder/encoder")
        self.decoder.save_weights(self.logdir+"/models/decoder/decoder")
        self.discriminator.save_weights(self.logdir+"/models/discriminator/discriminator")

        self.writer.flush()