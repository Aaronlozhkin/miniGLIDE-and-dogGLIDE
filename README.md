# miniGLIDE: Towards a Photo-Realistic Text to Image Model with a Reduction in Parameters

This repository is the attatched code to the low parameter GLIDE project for Rutgers CS:533 Natural Language Processing. This project was heavily inspired from [Nichol et. al. (2021) GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741).

Please click [here](https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/Final%20Report.pdf) to view the Final Report for our project.

## Sampling dogGLIDE or miniGLIDE

<a target="_blank" href="https://colab.research.google.com/github/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/SamplingDemo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

Please refer to the [Sampling Demo](https://github.com/Aaronlozhkin/miniGLIDE-and-dogGLIDE/blob/main/SamplingDemo.ipynb) for easy sampling of our pretrained models.

If you would like to sample the given models on your local machine manually, feel free to download them: [miniGLIDE](https://drive.google.com/file/d/1NXGr6wMeYOU98nxivZNtZBWXFRU42XyU/view?usp=share_link) and [dogGLIDE](https://drive.google.com/file/d/1NXGr6wMeYOU98nxivZNtZBWXFRU42XyU/view?usp=share_link). Place them in the [glide_model_cache](glide_model_cache) folder and run through the SamplingDemo notebook to try our models out.

## Training your own model
If you would like to try and train your own version of this model, please refer to [trainGLIDE.py](trainGLIDE.py)

Currently the script only supports training on the [sbucaptions dataset](https://www.cs.rice.edu/~vo9/sbucaptions/) and the [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). However, with some minor tunings of file structure any (image, caption) dataset will work.

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


