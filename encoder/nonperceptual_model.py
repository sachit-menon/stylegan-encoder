import numpy as np
import tensorflow as tf
from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K

def preprocess_input(images):
    return images/255.

def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


class NonperceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        # self.perceptual_model = None
        self.ref_img = None
        self.pixel_weight = None
        self.loss = None

    def build_nonperceptual_model(self, generated_image_tensor):
        # vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        # self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=2))
        # generated_img_features = self.perceptual_model(generated_image)

        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.pixel_weight = tf.get_variable('pixel_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.pixel_weight.initializer, self.pixel_weight.initializer])

        self.loss = tf.reduce_sum(tf.square(self.ref_img - generated_image))/256. #/10000. # tf.losses.mean_squared_error(self.ref_img,
                     #                            generated_image) # / 8289.0

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        # image_features = self.perceptual_model.predict_on_batch(loaded_image)

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.pixel_weight.shape)
        if len(images_list) != self.batch_size:
            pixel_space = list(self.pixel_weight.shape[1:])
            existing_pixel_shape = [len(images_list)] + pixel_space
            empty_pixel_shape = [self.batch_size - len(images_list)] + pixel_space

            existing_examples = np.ones(shape=existing_pixel_shape)
            empty_examples = np.zeros(shape=empty_pixel_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            loaded_image = np.vstack([loaded_image, np.zeros(empty_pixel_shape)])

        self.sess.run(tf.assign(self.pixel_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img, loaded_image))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss

