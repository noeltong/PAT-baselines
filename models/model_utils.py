from models.unet import UNet
from models.uformer import Uformer
from models.unetppp import UNet_3Plus

def get_arch(arch):
    if arch == 'UNet':
        model = UNet(dim=64)
    elif arch == 'Uformer':
        model = Uformer(img_size=128, embed_dim=64, win_size=8, token_projection='linear', token_mlp='leff', modulator=True)
    elif arch == 'Uformer_T':
        model = Uformer(img_size=128, embed_dim=16, win_size=8, token_projection='linear', token_mlp='leff', modulator=True)
    elif arch == 'Uformer_S':
        model = Uformer(img_size=128, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff', modulator=True)
    elif arch == 'Uformer_B':
        model = Uformer(img_size=128, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff', depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=1)
    elif arch == 'UNet_3P':
        model = UNet_3Plus()
    else:
        raise Exception("Arch error!")

    return model