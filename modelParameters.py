import torch as th
from PIL import Image
from IPython.display import display

def get_model_kwargs(prompts, model, options, device):
    # Initialize lists to store tokens and masks for each prompt.
    all_tokens = []
    all_masks = []
    
    for prompt in prompts:
        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Add tokens and masks to their respective lists.
        all_tokens.append(tokens)
        all_masks.append(mask)


    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            all_tokens, device=device
        ),
        mask=th.tensor(
            all_masks,
            dtype=th.bool,
            device=device,
        ),
    )

    return model_kwargs


def get_model_kwargs_classifier_free(prompts, model, options, device):
    # Initialize lists to store tokens and masks for each prompt.
    all_tokens = []
    all_masks = []
    
    for prompt in prompts:
        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Add tokens and masks to their respective lists.
        all_tokens.append(tokens)
        all_masks.append(mask)

    # Create the classifier-free guidance tokens (empty) for each prompt.
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            all_tokens + [uncond_tokens] * len(all_tokens), device=device
        ),
        mask=th.tensor(
            all_masks + [uncond_mask] * len(all_tokens),
            dtype=th.bool,
            device=device,
        ),
    )

    return model_kwargs

def get_model_kwargs_upsample(prompts, samples, model, options, device):
    # Initialize lists to store tokens and masks for each prompt.
    all_tokens = []
    all_masks = []
    
    for prompt in prompts:
        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Add tokens and masks to their respective lists.
        all_tokens.append(tokens)
        all_masks.append(mask)

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(

        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,
        
        tokens=th.tensor(
            all_tokens, device=device
        ),
        mask=th.tensor(
            all_masks,
            dtype=th.bool,
            device=device,
        ),
    )

    return model_kwargs

def show_images(batch: th.Tensor, brightness: float = 1.7, glide=False):
    """ Display a batch of images inline with adjustable brightness. """

    if glide:
        scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        display(Image.fromarray(reshaped.numpy()))
        return
    
    # Check if the batch has 3 or 4 dimensions
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    
    # Apply brightness adjustment
    batch = batch * brightness

    # Ensure the values are within the valid range for image display
    scaled = ((batch)*(127.5)).round().clamp(0,255).to(th.uint8).cpu()

    # Rearrange dimensions for image display
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])

    # Display the image
    display(Image.fromarray(reshaped.numpy()))