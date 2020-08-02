from models.model import Encoder, Decoder, Discriminator
from models.func_model import encoder, decoder

def get_ae(config):
    net_type = config["net_type"]
    assert net_type in ["mod", "fun"], "Network type not implemented"
    if net_type == "mod":
        return Encoder(config), Decoder(config)
    else:
        return encoder(config), encoder(config)

def get_discriminator(config):
    return Discriminator(config)

def get_ae_disc(config):
    return (*get_ae(config), get_discriminator(config))
    