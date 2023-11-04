# miniGLIDE and dogGLIDE

*miniGLIDE was created for Rutgers CS:533 Natural Language Processing. Special thanks to [Nichol et. al. (2021) GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741) for heavy project inspiration.*

[Aaron Lozhkin, Srinihar Bondalapati, Jinal Shah (2023) miniGLIDE: Towards a Photo-Realistic Text to Image Model With a Reduction in Parameters](https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/Final%20Report.pdf)

<p>
  <img src="https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/assets/23532191/7f802482-40e7-4fee-aaaf-b0dab1e18747" alt="Sample" width="400" />
  <img src= "https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/assets/23532191/b8b2d55e-f922-41fc-a357-ca7ec4183f10" width = "400"/>
</p>

miniGlide and dogGLIDE are text-to-image diffusion models implemented with classifier-free guidance. Both models consist of approximately 1.5 million parameters, a significant reduction compared to state-of-the-art diffusion models. Although outputs are not qualitatively comparable, these models were trained from scratch on a single GPU and show the capabilities of diffusion models with significant parameter reduction. 

## Requirements and Installation ##

This repository has been tested with python>=3.7

For installation, use git+ with pip in a [conda](conda.io) or local environment to download all dependencies

```
pip install git+https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE
```

Alternatively we provide a colab [Sampling Demo](https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/SamplingDemo.ipynb) for quick download and use of our models

<a target="_blank" href="https://colab.research.google.com/github/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/SamplingDemo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>


If you would like to sample the given models on your local machine manually, feel free to download them: [miniGLIDE](https://drive.google.com/file/d/1NXGr6wMeYOU98nxivZNtZBWXFRU42XyU/view?usp=share_link) and [dogGLIDE](https://drive.google.com/file/d/1NXGr6wMeYOU98nxivZNtZBWXFRU42XyU/view?usp=share_link). Place them in the [glide_model_cache](glide_model_cache) folder and run through the [Sampling Demo](https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/SamplingDemo.ipynb).

## miniGLIDE vs dogGLIDE

miniGLIDE is our generalist model, which attempts to create images from textual inference on most concepts within the English language. In practice, miniGLIDE struggles to accurately represent all aspects of a given piece of text. Most commonly, the color and only figments of a concept will appear in the final image. For this reason, we chose to also create dogGLIDE which focuses solely on images of dogs. In due part to the reduction in potential concepts, dogGLIDE is able to create realistic and unique images of dogs. In some cases, it performs better than the original [GLIDE](https://arxiv.org/abs/2112.10741) although the comparison is not entirely fair.

miniGLIDE was trained from empty weights on the [sbucaptions dataset](https://www.cs.rice.edu/~vo9/sbucaptions/) and dogGLIDE was trained on the [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## Classifier Free Guidance

When sampling with these diffusion models, one parameter to adjust is the  **guidance scale**. This informs the model how far to push its internal representation of the image towards the specified text. It is not always the case that higher guidance scales result in better images.

<p align="center">
  <img src="https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/assets/23532191/7cfadea1-9682-4fea-be5a-73c9c55ff071" />
</p>

## Primary Architecture Changes From GLIDE

The main reduction in parameter size was achieved from the following, in decreasing order of significance:

- Changing the number of channels from 192 to 96
- Changing the number of channels per attention head from 64 to 32
- Changing the number of attention heads from 4 to 1

These changes by themselves reduced the parameter count significantly. 

All other aspects of the diffusion architecture was kept in place from the hybrid loss model to the denoising UNET.

## Training your own model

Training your own diffusion model can be done with [trainGLIDE.py](trainGLIDE.py). This script officially supports the [sbucaptions dataset](https://www.cs.rice.edu/~vo9/sbucaptions/) and [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), but with some minor tunings of file structure any (image, caption) dataset will work.

### Parameters:

- `--data-dir`: The directory where your input data is located. The script currently supports the sbucaptions dataset and the images folder of the Stanford Dog Dataset. Default is 'Data/sbucaptions'.
- `--image-limit`: The maximum number of images to load from the sbucaptions dataset. Default is 100,000. All images are always loaded from the dog dataset.
- `--dogs`: Sets the dataset to be the Stanford Dog Dataset. Default is False.
- `--checkpoint-dir`: The directory to save model checkpoints during training. Default is 'checkpoints/'.
- `--checkpoint-time`: The number of seconds in between checkpoint saves. Default is 3600 (1 hour).
- `--continue-train`: A path to a model checkpoint to continue training from. Ensure that this model's parameters are the same as specified with other tags or with --defaults. Default is None.
- `--num-channels`: The number of channels in the UNET model. Must be divisible by num_head_channels. Default is 96.
- `--num-head-channels`: The number of attention heads. Default is 32.
- `--num-res-block`: The number of residual channels. Default is 3. Not recommended to wildly change this value from the default. Low Res Blocks produce poor quality output.
- `--xf-width`: The width of the transformer model. Default is 512.
- `--xf-layers`: The number of layers in the transformer model. Default is 16.
- `--xf-heads`: The number of attention heads in the transformer model. Default is 16.
- `--defaults`: Predefines a model shape and size with the default model and training arguments above. Overrides any other arguments passed in.
- `--help`: Shows all arguments and their defualt values. Refer here for finer tunining of the model's parameters and training.


