import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
from encoder.nonperceptual_model import NonperceptualModel

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

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=2000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, randomize_noise=args.randomize_noise)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)
    # nonperceptual_model = NonperceptualModel(args.image_size, batch_size=args.batch_size)
    # nonperceptual_model.build_nonperceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        perceptual_model.set_reference_images(images_batch)
        # nonperceptual_model.set_reference_images(images_batch)
        
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, learning_rate=args.lr)
        # op = nonperceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, learning_rate=args.lr)
        
        pbar = tqdm(op, leave=False, total=args.iterations)
        min_loss = np.inf
        img = None
        for i,per_loss,reg_loss, loss in pbar:
            # Generate images from found dlatents and save them
            if(loss < min_loss and i>0.4*args.iterations):
                min_loss = loss

                generated_images = generator.generate_images()
                generated_dlatents = generator.get_dlatents()
                for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
                    img = PIL.Image.fromarray(img_array, 'RGB')

                print('\n'+' '.join(names)+' Per/Reg/Total Loss: [{0:.2f},{1:.2f},{2:.2f}]'.format(per_loss,reg_loss,loss)+'<-- BEST')
            else:
                print('\n'+' '.join(names)+' Per/Reg/Total Loss: [{0:.2f},{1:.2f},{2:.2f}]'.format(per_loss,reg_loss,loss))

            if(i%100 == 0 and img is not None):
                img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
                np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)
        print(' '.join(names), ' loss:', loss)
        img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
        np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        # # Generate images from found dlatents and save them
        # generated_images = generator.generate_images()
        # generated_dlatents = generator.get_dlatents()
        # for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
        #     img = PIL.Image.fromarray(img_array, 'RGB')
        #     img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
        #     np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
