import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch as th
import torch.nn as nn
import torch.optim as optim


from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt
import random

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

from diffusionHelp import *
from modelParameters import *

import argparse
import sys
import os
import time
from tqdm import tqdm

# Import sbucaptions dataset

class dogDataset(Dataset):
    def __init__(self, root_dir, image_limit):
        self.root_dir = root_dir
        self.image_data = []
        self.caption_data = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        for folder in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, folder)):
                name = folder.split('-')[1]
                for file in os.listdir(os.path.join(root_dir, folder)):
                    if len(self.image_data) > image_limit:
                            break
                    if file.endswith('.jpg'):
                        self.image_data.append(root_dir + folder + '/' + file)
                        self.caption_data.append(str(name))

    def generate_prompts(self, num_prompts):
        return [self.caption_data[random.randint(0, len(self.caption_data)-1)] for i in range(num_prompts)]
    
    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_data[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        caption_info = self.caption_data[idx]

        return image, caption_info
    

class SbuCaptionsDataset(Dataset):
    def __init__(self, root_dir, image_limit):
        self.root_dir = root_dir
        self.image_data = []
        self.caption_data = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        print("Loading sbucaptions Dataset: ")
        subdirs = [x[0] for x in os.walk(self.root_dir)]

        for subdir in tqdm(subdirs, total=len(subdirs)):
            json_files = [f for f in os.listdir(subdir) if f.endswith('.json') and 'stats' not in f]
            
            for json_file in json_files:

                # if len(self.caption_data)+1 % 10000 == 0:
                #     print("Loaded " + str(len(self.caption_data)) + " images")

                if len(self.caption_data) > image_limit:
                    break

                with open(os.path.join(subdir, json_file), 'r') as f:
                    data = json.load(f)
                    if data['status'] == 'success' and data['error_message'] in [None, 'null']:
                        self.image_data.append(os.path.join(subdir, data['key'] + '.jpg'))
                        self.caption_data.append({
                            'caption': data['caption'],
                        })

    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_data[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        caption_info = self.caption_data[idx]['caption']

        return image, caption_info

last_save_time = time.time()

def run_epoch(dataloader, model, diffusion, optimizer, criterion, scheduler, 
              options, device, batch_size, checkpoint_dir, checkpoint_time, checkpoints, epoch, is_vlb=True, delcheck=False, is_training=False):

    epoch_loss = 0

    if is_training:
        model.train(True)
    else:
        model.eval()

    for idx, (x_start,caption) in enumerate(dataloader):

        if x_start.shape[0] != batch_size:
            continue

        x_start = x_start.to(device)

        model_kwargs = get_model_kwargs(list(caption), model, options, device)

        noise = torch.randn(x_start.shape).to(x_start.device)

        # sample uniformly from {1 ... T}

        time_steps = torch.randint(0, max(diffusion.timestep_map), (batch_size,), device=device)

        x_t = diffusion.q_sample(x_start, time_steps, noise=noise)
        true_mean, _, true_logvar = diffusion.q_posterior_mean_variance(x_start, x_t, time_steps)

        out = p_epsilon_var(diffusion,
            model,
            x = x_t,
            t = time_steps,
            model_kwargs=model_kwargs)
        
        epsilon_theta = out['eps']
        
        loss_simple = criterion(noise, epsilon_theta)

        if is_vlb:

            # Now, we will implement the solutions by Dhariwal et al. (2021) to calculate the loss of the variational lower bound
            # KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            # KL Divergence between two gaussians: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

            loss_vlb = 0.5 * (-1.0 + out['log_variance'] - true_logvar + 
                            th.exp(true_logvar - out['log_variance']) + 
                            ((true_mean - out['mean']) ** 2) * th.exp(-out['log_variance']))

            # add the variational lower bound loss * lambda=0.001 to loss_simple

            loss = loss_simple + th.mean(loss_vlb.flatten()) * (1/1000)
        
        else:
            loss = loss_simple

        if is_training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        global last_save_time
        current_time = time.time()
        if current_time - last_save_time >= checkpoint_time:
            # Save the model
            save_path = f"{checkpoint_dir}/dog_model_epoch{epoch}_time{current_time}.pt"
            print("saving Model")
            th.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'epoch': epoch+1}, save_path)
            checkpoints.append(save_path)

            if delcheck and len(checkpoints) >= 3:
                file = checkpoints[0]
                if os.path.exists(file):
                    os.remove(file)
                    checkpoints.pop(0)
                else:
                    print("Checkpoint file not deleted. File does not exist")

            
            # Update the last_save_time variable
            last_save_time = current_time

        epoch_loss += (loss.detach().item() / batch_size)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


def main():

    # Parse arguments

    parser = argparse.ArgumentParser(description='Trains a blank version of the GLIDE model. Requires a dataset of images and captions. Pass in -h for help.')

    # define the directory arguments
    parser.add_argument('--data-dir', type=str, default='Data/sbucaptions', help='The directory of your input data. ' + 
                        'Currently only supports the sbucaptions dataset and the images folder of the Stanford Dog Dataset.')
    parser.add_argument('--image-limit', type=int, default=100000, help='The maximum number of images to load from the sbucaptions dataset. '
                        + 'Default is 100,000. All images are always loaded from dog dataset.')
    parser.add_argument('--dogs', action='store_true', help='Sets the dataset to be the Stanford Dog Dataset. Default is False.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='The directory to save model checkpoints during training. Default is checkpoints/.')
    parser.add_argument('--checkpoint-time', type=int, default=3600, help='The number of seconds in between checkpoint saves. Default is 3600 (1 hour).')
    parser.add_argument('--continue-train', type=str, default=None, help='A path to a model checkpoint to continue training from. ' +
                        'Ensure this models parameters are the same as specified with other tags or with --defaults. Default is None.')
    parser.add_argument('--delcheck', action='store_true', help='If true then progressively deletes older checkpoints of the model. Saves only the three most recent. Default is False')

    # define the training arguments
    parser.add_argument('--epochs', type=int, default=100, help='The directory of your input data. Currently only supports the sbucaptions dataset.')
    parser.add_argument('--batch-size', type=int, default=12, help='The batch size of your data. Default is 8. Be wary of memory constraints.')
    parser.add_argument('--timesteps', type=int, default=100, help='The number of diffusion timesteps to run the model for. Default is 100.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='The learning-rate of the model. Default is 1e-4. Utilized with the Adam optimizer.')

    # define the model arguments
    parser.add_argument('--num-channels', type=int, default=96, help='The number of channels in the UNET model. Must be divisible by num_head_channels. Default is 96.')
    parser.add_argument('--num-head-channels', type=int, default=32, help='The number of attention heads. Default is 32.')
    parser.add_argument('--num-res-block', type=int, default=3, help='The number of residual channels. Default is 3.' + 
                        'Not recommended to wildly change this value from the default. Low Res Blocks produce poor quality output.')
    parser.add_argument('--xf-width', type=int, default=512, help='The width of the transformer model. Default is 512.')
    parser.add_argument('--xf-layers', type=int, default=16, help='The number of layers in the transformer model. Default is 16.')
    parser.add_argument('--xf-heads', type=int, default=8, help='The number of attention heads in the transformer model. Default is 16.')
    parser.add_argument('--caption', type=str, default="chihuahua", help='Specifies the caption to use for sampling. Only relevant if passed with --verbose. Default is chihuahua.')
   
    # define boolean flags for default model parameters
    parser.add_argument('--defaults', action='store_true', help='Predefines a model shape and size with the default model and training arguments above. Overrides any other arguments passed in.')
    parser.add_argument('--no-vlb', action='store_true', help='Forces the model to only train on simple loss and drops the variational lower bound. Default is False')

    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    print(f'Using device {device}', flush=True)

    if not has_cuda:
        print("WARNING: CUDA is highly recommended for this model. Expect extremely slow performance on CPU.")

    options = model_and_diffusion_defaults()

    if args.defaults:
        print("Using default model parameters.", flush=True)
        options['use_fp16'] = False
        options['diffusion_steps'] = 100
        options['num_channels'] = 96
        options['num_head_channels'] = 32
        options['num_res_blocks'] = 3
        options['xf_width'] = 512
        options['xf_layers'] = 16
        options['xf_heads'] = 8

        epochs = 100
        batch_size = 8
        learning_rate = 1e-4

    else:

        if args.num_channels % args.num_head_channels != 0:
            sys.exit('num_channels must be divisible by num_head_channels.')

        options['use_fp16'] = False
        options['diffusion_steps'] = args.timesteps
        options['num_channels'] = args.num_channels
        options['num_head_channels'] = args.num_head_channels
        options['num_res_blocks'] = args.num_res_block
        options['xf_width'] = args.xf_width
        options['xf_layers'] = args.xf_layers
        options['xf_heads'] = args.xf_heads

        epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate

    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    model.to(device)
    print('total base parameters', sum(x.numel() for x in model.parameters()), flush=True)

    # Usage example

    if args.dogs:
        root_dir = args.data_dir
        dataset = dogDataset(root_dir, image_limit=args.image_limit)
        print("Images in dog_dataset: ", len(dataset), flush=True)
    else:
        root_dir = args.data_dir
        dataset = SbuCaptionsDataset(root_dir, image_limit=args.image_limit)
        print("Images in sbu_captions_dataset: ", len(dataset), flush=True)
    
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print("Created checkpoint directory at: " + str(args.checkpoint_dir), flush=True)

    checkpoint_time = args.checkpoint_time
    checkpoint_dir = args.checkpoint_dir

    # create `DataLoader`
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    epoch_start = 0

    if args.continue_train:
        optimizer.load_state_dict(th.load(args.continue_train['optimizer']))
        model.load_state_dict(th.load(args.continue_train['model']))
        epoch_start = args.continue_train['epoch']

        try:
            scheduler.load_state_dict(th.load(args.continue_train['scheduler']))
        except:
            print("No scheduler found. Continuing without one.", flush=True)     
    
    criterion = nn.MSELoss()

    loss_list = []
    checkpoints = []

    # begin training
    for epoch in range(epoch_start, epochs):
        loss_train, lr_train = run_epoch(train_dataloader, model, diffusion, optimizer, criterion, scheduler,
                                          options, device, batch_size, checkpoint_dir, checkpoint_time, checkpoints=checkpoints, epoch, is_vlb=args.no_vlb, delcheck=args.delcheck, is_training=True)
        loss_list.append(loss_train)
        scheduler.step()
        
        print('Epoch[{}/{}] | loss train:{:.6f} | lr:{:.6f}'
                .format(epoch+1, epochs, loss_train, lr_train), flush=True)
        
    # Save the final model
    save_path = os.path.join(str(checkpoint_dir), 'dog_model_iLAB.pt')
    print("Saving Final Model to: " + str(save_path), flush=True)
    th.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'epoch': epoch+1}, save_path)
    

if __name__ == '__main__':
    main()