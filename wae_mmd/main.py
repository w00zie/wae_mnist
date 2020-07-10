from train import Train
import os

# Logging only (W)arnings and (E)rrors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

BATCH_SIZE = 100
EPOCHS = 51
train = Train(batch_size=BATCH_SIZE, 
              epochs=EPOCHS,
              h_dim=32,
              z_dim=10,
              conv_kernel_size=(5,5),
              kernel_init="TruncatedNormal",
              sigma_z=1.,
              enc_dec_lr=1e-3, 
              lmbda=10.)
train.train()
