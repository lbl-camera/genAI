
# Synthetic Scientific Image Generation

This repository provides a comprehensive exploration of synthetic image generation using multiple generative model architectures: **DCGAN**, **StyleGAN2-ADA**, **DALL-E 2 & 3**, and **Diffusion-based models** (e.g., Stable Diffusion, ControlNet, InstructPix2Pix, etc.).

🧪 This work is supported by the paper:  
**"Synthetic Scientific Image Generation with VAE, GAN, and Diffusion Model Architectures"**  
📄 [Read the full article](https://www.mdpi.com/2313-433X/11/8/252)

---

## 📁 Contents

| Notebook | Description |
|----------|-------------|
| `dcgan_training.ipynb` | DCGAN training notebook. Users can train a DCGAN model on their **own dataset** (we do not distribute the dataset used in the original experiments). |
| `stylegan2_training_cli.md` | Command-line instructions to train StyleGAN2-ADA using the official NVIDIA repo. |
| `dalle_generation.ipynb` | Inference using DALL-E 2 and 3 APIs, including inpainting (edit), variation, and text-to-image generation. |
| `diffusion_models_inference.ipynb` | Inference using various diffusion-based models (ControlNet, Stable UnCLIP, InstructPix2Pix, LEDITS++, DiffEdit). Includes multiple examples. |

---

## Supported Models

| Model Type | Models Used |
|------------|-------------|
| GANs | DCGAN, StyleGAN2-ADA |
| Transformer-based | DALL-E 2, DALL-E 3 |
| Diffusion-based | Stable Diffusion, ControlNet, InstructPix2Pix, LEdit++, DiffEdit |

---

## Sample Datasets

For **DALL-E** and **diffusion-based** model notebooks, we include sample images from 3 scientific datasets:

- **CMC** – MicroCT images of fiber beads  
- **ROCKS** – MicroCT images of rock sediments  
- **ROOTS** – High-resolution RGB images of plant roots

Each dataset contains:
- 1 input image
- 3 generated images per input
- 1 set of outputs per model (i.e., each model generates 3 outputs per image)

---

## DCGAN Notebook

- `dcgan_training.ipynb` provides a complete implementation of a DCGAN training loop.
- 🛑 **Note**: You must use your own training data. We do **not** provide the training set used in our original experiments.

---

## StyleGAN2-ADA Training (Command Line)

Training StyleGAN2-ADA is done via the [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository.

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

## Contact

For questions, feel free to open an issue or contact us through the paper correspondence info.
