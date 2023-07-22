from models.unet import UNet
from models.uformer import Uformer

def get_arch(opt):
    arch = opt.arch
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'Uformer':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_T':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_S':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_B':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=opt.dd_in)
    else:
        raise Exception("Arch error!")

    return model_restoration