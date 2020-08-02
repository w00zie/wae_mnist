config_mmd = {
    "h_dim": 16,
    "z_dim": 10,
    "num_epochs": 11,
    "batch_size": 100,
    "conv_kernel_size": [5,5],
    "net_type": "mod",
    "kernel_init": "TruncatedNormal",
    "ae_lr": 1e-3,
    "ae_dec_steps": 10000,
    "ae_dec_rate": 0.95,
    "sigma_z": 1.0,
    "lambda": 10,
}