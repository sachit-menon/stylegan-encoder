import logging
logging.getLogger('tensorflow').disabled = True

import os
from pathlib import Path
from datetime import datetime
import glob

file_path = Path().resolve()
print("Current Path: " + str(file_path))

last_edit,last_file = max([(os.stat(filename).st_mtime,filename) for filename in Path().glob('**/*.py')])
last_edit = datetime.fromtimestamp(last_edit)
current_time = datetime.now()
diff_time = current_time-last_edit
print("Last python file (" + str(last_file) + ") was modified " + str(diff_time.total_seconds())+" seconds ago")

import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from tensorflow.train import cosine_decay_restarts
from matplotlib import pyplot as plt
import scipy.misc
from SR.generator import Generator
from SR.optimizer import SROptimizer
from tqdm import trange

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

def get_args():
    parser = argparse.ArgumentParser(description='Style-Gan Super Resolution')
    parser.add_argument('-i', '--input_files', help='Input files')
    parser.add_argument('-o', '--output_dir', default="generated_images/", help='Directory for storing output images and latents')
    parser.add_argument('--loss_dir', default="Loss/", help='Directory for storing loss logs')
    parser.add_argument('--n_init', default=1, help='Number of initializations for encoder', type=int)
    parser.add_argument('--img_size', default=64, help='Size to rescale to', type=int)
    parser.add_argument('--loss', default='1.0*L2', help='Loss function to use')
    parser.add_argument('-LIN','--layersIN', default=[3,6,9], nargs='+',help='Which VGG-ImageNet layers to use',type=int)
    parser.add_argument('-LF','--layersF', default=[3,6,9], nargs='+',help='Which VGG-Face layers to use',type=int)
    parser.add_argument('--lr', default=1., help='Learning rate', type=float)
    parser.add_argument('--optimizer', default='SGD', help='Which optimizer to use')
    parser.add_argument('--cosine_cycle', default=None, help='Whether to use cosine annealing',type=int)
    parser.add_argument('--steps', default=1500, help='Number of gradient-descent steps', type=int)
    parser.add_argument('--mask_type', default=None, help='Whether to weight the pixels differently with a face mask')

    args = parser.parse_args()
    return args

def plot_loss(losslist,best_losses,loss_dir):
    losses = np.array(losslist)
    axis = np.array(range(len(losses)))
    label = 'Loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, losses, label="Loss",zorder=1)
    plt.scatter(axis[best_losses], losses[best_losses], s=5, c='r', label="Best",zorder=2)
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(loss_dir/'loss.pdf')
    plt.close(fig)

def main():
    args = get_args()

    ref_images = Path().glob(args.input_files)
    # ref_images = list(ref_images)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    loss_dir = Path(args.loss_dir)
    loss_dir.mkdir(exist_ok=True)

    # Load StyleGAN
    print("Loading StyleGAN")
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    print("Loaded StyleGAN")

    print("Creating Generator")
    G = Generator(Gs_network, args)
    G.reset_latent()
    print("Created Generator")
    print("Creating Optimizer")
    opt = SROptimizer(G, args)
    print("Created Optimizer")
    print("Building Loss")
    opt.build_loss()
    print("Built Loss")

    losses=[]
    best_losses=[]

    for image in ref_images:
        print(f"Working on {image.name}")
        opt.set_reference_image(image)

        image_mask = opt.image_mask.eval()[0]
        image_mask = np.clip(255*image_mask,0,255).astype('uint8')
        image_mask = PIL.Image.fromarray(image_mask)
        image_mask.save(output_dir/'mask.png')

        t = trange(args.steps)

        min_loss = np.inf
        best_img = None
        best_idx = 0
        best_list = [np.inf,np.inf,np.inf]
        cos_iter = 0
        cos_cycle = args.cosine_cycle

        for i in t:
            current_lr = args.lr
            if(cos_cycle is not None):
                current_lr = 1e-5 + 0.5 * (args.lr - 1e-5) * \
                 (1. + np.cos(cos_iter * np.pi / cos_cycle))
                cos_iter += 1
                if(cos_iter == cos_cycle):
                    cos_iter = 0
                    cos_cycle = 2*cos_cycle
            loss_list,loss = opt.step(lr=current_lr)
            losses.append(loss)
            curr_str = ' '.join([f"{x[0]}: {x[1]:.3f}/{y:.3f}" for x,y in zip(loss_list,best_list)])
            curr_str += f", TOTAL: {loss:.3f}/{min_loss:.3f}, BEST_IDX: {best_idx}, LR: {current_lr:.5f}"

            if(loss < min_loss):
                min_loss = loss
                best_losses.append(i)
                best_idx = i
                best_list = [x[1] for x in loss_list]
                if(i>=0.1*args.steps):
                    generated_image = G.generate_images()
                    generated_latent = G.latent.eval()
                    best_img = PIL.Image.fromarray(generated_image[0], 'RGB')
                    best_latent=generated_latent

            t.set_description(curr_str)

            if(i%100 == 0):
                if(best_img is not None):
                    best_img.save(output_dir / image.name, 'PNG')
                    np.save(output_dir / image.stem, best_latent)
                plot_loss(losses,best_losses,loss_dir)

if __name__ == "__main__":
    main()
