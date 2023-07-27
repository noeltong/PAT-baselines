# Baselines for PAT image enhancement and reconstruction

## Related Works


| Implemented | Architecture | Paper Title | Citation |
|-------------|:------------:|:-----------:|----------|
|    ✅      | UNet         |  U-Net: Convolutional Networks for Biomedical Image Segmentation   |    [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6364fdaa0a0eccd823a779fcdd489173f938e91a%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/U-Net%3A-Convolutional-Networks-for-Biomedical-Image-Ronneberger-Fischer/6364fdaa0a0eccd823a779fcdd489173f938e91a)      |
|    ✅      |   Uformer    |   Uformer: A General U-Shaped Transformer for Image Restoration     |  [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2835951fabf12804e17d5a525b2be2bee70e7910%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Uformer%3A-A-General-U-Shaped-Transformer-for-Image-Wang-Cun/2835951fabf12804e17d5a525b2be2bee70e7910)   |
|             |   UNet++   |   UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation   |  [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F42b0a8f757e45462e627e57f9af7e9849dcdacdf%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/UNet%2B%2B%3A-Redesigning-Skip-Connections-to-Exploit-in-Zhou-Siddiquee/42b0a8f757e45462e627e57f9af7e9849dcdacdf)   |
|    ✅       |  UNet 3+    |  UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation   |   [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0b444f74dd9cc06c2833dd15f9258ef5e169e6ea%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/UNet-3%2B%3A-A-Full-Scale-Connected-UNet-for-Medical-Huang-Lin/0b444f74dd9cc06c2833dd15f9258ef5e169e6ea)   |
|             |   FD UNet    |   Fully Dense UNet for 2-D Sparse Photoacoustic Tomography Artifact Removal   |    [![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F06e39ff951f8a137ba871a3f62de19bef1c128a8%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Fully-Dense-UNet-for-2-D-Sparse-Photoacoustic-Guan-Khan/06e39ff951f8a137ba871a3f62de19bef1c128a8)    |

## Train

Run

```shell
torchrun --nproc_per_node=4 main.py --mode train --config configs/unet3p.py
```

## Evaluation