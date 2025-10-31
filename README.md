
# ApexGenAI - Synthetic Scientific Image Generation

This repository provides a comprehensive exploration of synthetic image generation using multiple generative model architectures: **DCGAN**, **StyleGAN2-ADA**, **DALL-E 2 & 3**, and **Diffusion-based models** (e.g., Stable Diffusion, ControlNet, InstructPix2Pix, etc.).

This work is supported by the paper:  
**"Synthetic Scientific Image Generation with VAE, GAN, and Diffusion Model Architectures"**  
📄 [Read the full article](https://www.mdpi.com/2313-433X/11/8/252)

---

## Contents

| Notebook | Description |
|----------|-------------|
| `dcgan_training.ipynb` | DCGAN training notebook. Users can train a DCGAN model on their **own dataset** (we do not distribute the dataset used in the original experiments, only 3 samples for each type of data which are presented below). |
| `stylegan2_training_cli.md` | Command-line instructions to train StyleGAN2-ADA using the official NVIDIA repo. |
| `dalle_generation.ipynb` | Inference using DALL-E 2 and 3 APIs, including inpainting (edit), variation, and text-to-image generation. |
| `diffusion_models_inference.ipynb` | Inference using various diffusion-based models (ControlNet, Stable UnCLIP, InstructPix2Pix, LEDITS++, DiffEdit). Includes multiple examples. |

These notebooks can be used and run using 1 GPU on google colab. We also provide a link to the data: 



---

## Supported Models

| Model Type | Models Used |
|------------|-------------|
| GANs | DCGAN, StyleGAN2-ADA |
| Transformer-based | DALL-E 2, DALL-E 3 |
| Diffusion-based | Stable Diffusion (Stable unCLIP), ControlNet, InstructPix2Pix, LEdit++, DiffEdit |

---

## Sample Datasets

For **DALL-E** and **diffusion-based** model notebooks, we include sample images from 3 scientific datasets:

- **`CMC/`** – Ceramic Matrix Composites (CMCs) high-resolution imaging, achieved using synchrotron X-ray radiation composed of a class of materials engineered to enhance toughness and high-temperature performance compared to monolithic ceramics.
- **`ROCKS/`** – MicroCT scans from samples containing large sediment grains from the Hanford DOE contaminated nuclear site.
- **`ROOTS/`** – Plant root images: slices scanned by an automated robotic system called EcoBOT that enables high-throughput scanning of plants in hydroponic systems known as EcoFABs.

Each dataset contains:
- 3 input images
- 3 generated images (one image generated per input)

---


## DCGAN Notebook

- `dcgan_training.ipynb` provides a complete implementation of a DCGAN training loop.
- **Note**: You must use your own training data. We do **not** provide the training set used in our original experiments.

---

## StyleGAN2-ADA Training (Command Line)

Training StyleGAN2-ADA is done via the [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository. Users will be subject to the terms and conditions of the NVLabs StyleGAN2-ADA software. 

### Step-by-step:

1. **Set up environment**  
   Follow the official repo's instructions to create and activate a Python environment.

2. **Prepare your dataset**  
   Convert your image dataset to the `.zip` format:
   ```bash
   python dataset_tool.py --source=YOUR_IMAGE_FOLDER --dest=YOUR_OUTPUT_FOLDER/dataset.zip
   ```

3. **Train StyleGAN2-ADA with augmentation**  
   ```bash
   python train.py      --outdir=OUTPUT_DIRECTORY      --data=PATH_TO_ZIPPED_DATASET      --gpus=4      --mirror=1      --augpipe=bgcfnc      --resume=PATH_TO_CHECKPOINT_PKL  # optional
   ```

See the [official StyleGAN2-ADA tutorial](https://github.com/NVlabs/stylegan2-ada-pytorch) for full details.

---

## DALL-E 2 and 3 inference notebooks

- `dalle_generation.ipynb` includes examples for:
  - **Text-to-image** generation (DALL-E 2 and 3)
  - **Image variation** (DALL-E 2)
  - **Inpainting / Edit** mode (DALL-E 2)

Requires API access to OpenAI's services.

---

## Diffusion Models Notebook

- `diffusion_models_inference.ipynb` uses multiple models to perform inference:
  - **ControlNet**
  - **Stable UnCLIP**
  - **LEDITS++**
  - **DiffEdit**
  - **InstructPix2Pix**

- Each model generates 3 outputs for each input image from the 3 sample datasets (CMC, ROCKS, ROOTS).


## Metrics Calculation Notebook

- `calculate_metrics.ipynb` calculates all metrics for all models including (SSIM, LPIPS, FID and CLIP Score for text-guided models). 

---

## Installation & Dependencies

We recommend using separate environments for each model class. For example:

- DCGAN: standard PyTorch + torchvision
- StyleGAN2: follow official repo setup
- DALL-E: requires OpenAI API and `openai` Python package
- Diffusion: install `diffusers`, `transformers`, `xformers`, etc.

Each notebook includes cells to install all necessary dependencies. 

---

## Citation

If you use this repository or datasets in your research, please cite:

> Zineb Sordo, et al. (2024). *Synthetic Scientific Image Generation with VAE, GAN, and Diffusion Model Architectures*. MDPI Journal of Imaging. [https://www.mdpi.com/2313-433X/11/8/252](https://www.mdpi.com/2313-433X/11/8/252)

---
## Copyright Notice

ApexGenAI Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


---

## License Agreement

ApexGenAI Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof,
in binary and source code form.


---

## Contact

For questions, feel free to open an issue or contact us through the paper correspondence info.
