import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial

class Generator:
    def __init__(self, model, args):
        self.model = model

        self.n_init = args.n_init

        gauss_mean, _ = np.load("gaussian_fit.npy");
        self.latent = tf.get_variable("latent",shape=(self.n_init,18,512),dtype='float32')
        self.initial_latent = np.tile(gauss_mean,(self.n_init,18,1))

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.raw_generator_output = self.model.components.synthesis.get_output_for(self.latent)
        self.generated_image = tflib.convert_images_to_uint8(self.raw_generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

    def reset_latent(self):
        self.set_latent(self.initial_latent)

    def set_latent(self, latent):
        assert (latent.shape == self.latent.shape)
        self.sess.run(tf.assign(self.latent, latent))

    def get_latent(self):
        return self.sess.run(self.latent)

    def generate_images(self, latent=None):
        if latent:
            self.set_latent(latent)
        return self.sess.run(self.generated_image_uint8)
