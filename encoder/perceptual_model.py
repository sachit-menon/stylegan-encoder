import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    # preprocessed_images = preprocess_input(loaded_images)
    return loaded_images #preprocessed_images


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None

        self.ref_img_features1 = None
        self.features_weight1 = None

        self.ref_img_features2 = None
        self.features_weight2 = None

        self.ref_img = None

        self.nonperceptual_loss = None
        self.perceptual_loss = None
        self.reg_loss = None
        self.loss = None

        self.vars_to_optimize = None

        self.latent_avg = np.load("latent_avg.npy")
        self.gauss_mean, self.gauss_scale = np.load("gaussian_fit.npy")

    def build_perceptual_model(self, generated_image_tensor):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, [vgg16.layers[self.layer].output, vgg16.layers[6].output, vgg16.layers[3].output])
        generated_image = tf.image.resize_images(generated_image_tensor,
                                                 (self.img_size, self.img_size), method=2)

        preprocessed_generated_image = preprocess_input(generated_image)
        generated_img_features = self.perceptual_model(preprocessed_generated_image)

        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features[0].shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features[0].shape,
                                               dtype='float32', initializer=tf.initializers.zeros())

        self.ref_img_features1 = tf.get_variable('ref_img_features1', shape=generated_img_features[1].shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight1 = tf.get_variable('features_weight1', shape=generated_img_features[1].shape,
                                               dtype='float32', initializer=tf.initializers.zeros())

        self.ref_img_features2 = tf.get_variable('ref_img_features2', shape=generated_img_features[2].shape,
                                                 dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight2 = tf.get_variable('features_weight2', shape=generated_img_features[2].shape,
                                                dtype='float32', initializer=tf.initializers.zeros())


        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                       dtype='float32', initializer=tf.initializers.zeros())

        self.sess.run([self.features_weight.initializer, self.features_weight.initializer])

        self.sess.run([self.features_weight1.initializer, self.features_weight1.initializer])

        self.sess.run([self.features_weight2.initializer, self.features_weight2.initializer])


        self.perceptual_loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
                                                 self.features_weight * generated_img_features[0]) / 82890.0 \
                               + tf.losses.mean_squared_error(self.features_weight1 * self.ref_img_features1,
                                                            self.features_weight1 * generated_img_features[1]) / 82890.0 \
                               + tf.losses.mean_squared_error(self.features_weight2 * self.ref_img_features2,
                                                            self.features_weight2 * generated_img_features[2]) / 82890.0




        self.nonperceptual_loss = 20.0*tf.reduce_mean(tf.abs(self.ref_img/255. - generated_image/255.))

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        preprocessed_image = preprocess_input(loaded_image)
        image_features = self.perceptual_model.predict_on_batch(preprocessed_image)

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        weight_mask1 = np.ones(self.features_weight1.shape)
        weight_mask2 = np.ones(self.features_weight2.shape)

        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.features_weight1, weight_mask1))
        self.sess.run(tf.assign(self.features_weight2, weight_mask2))


        self.sess.run(tf.assign(self.ref_img_features, image_features[0]))
        self.sess.run(tf.assign(self.ref_img_features1, image_features[1]))
        self.sess.run(tf.assign(self.ref_img_features2, image_features[2]))

        self.sess.run(tf.assign(self.ref_img, loaded_image))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        self.vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]

        learning_rate_ph = tf.placeholder(tf.float32, shape=[])
        current_lr = learning_rate

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ph)

        # self.reg_loss_pos = tf.reduce_sum(tf.square(tf.nn.relu(self.vars_to_optimize[0])))/200.0
        # self.reg_loss_neg = tf.reduce_sum(tf.square(tf.nn.relu(-self.vars_to_optimize[0])))/200.0

        corrected_latent = self.vars_to_optimize[0] - 0.8*tf.nn.relu(self.vars_to_optimize[0])

        self.reg_loss = tf.reduce_sum(tf.square(corrected_latent-self.gauss_mean)/self.gauss_scale)/512.0

        lambda_per = 1
        lambda_nonper = 0
        lambda_reg = 0.5

        self.loss = lambda_nonper*self.nonperceptual_loss + lambda_per*self.perceptual_loss + lambda_reg*self.reg_loss

        min_op = optimizer.minimize(self.loss, var_list=[self.vars_to_optimize])

        for i in range(iterations):
            # if(i==100): current_lr = 1
            # elif(i<500): current_lr = 0.5
            # elif(i<1000): current_lr = 0.25
            # else: current_lr = 0.1

            current_lr=learning_rate

            _, per_loss, reg_loss, loss = self.sess.run([min_op, self.perceptual_loss, self.reg_loss, self.loss],feed_dict={learning_rate_ph: current_lr})
            
            # print(vars_to_optimize.eval())
            # print((tf.square(corrected_latent-self.gauss_mean)/self.gauss_scale).eval())

            yield i,per_loss,reg_loss,loss

