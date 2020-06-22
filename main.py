from train import Train
import os
# Logging only (W)arnings and (E)rrors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BATCH_SIZE = 100
EPOCHS = 101
train = Train(batch_size=BATCH_SIZE, 
              epochs=EPOCHS,
              h_dim=64,
              z_dim=10,
              conv_kernel_size=(5,5),
              kernel_init="TruncatedNormal",
              disc_units=512,
              disc_lr=3e-4,
              sigma_z=1.,
              enc_dec_lr=5e-3, 
              lmbda=5)
train.train()
