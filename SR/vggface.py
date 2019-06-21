import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from PIL import Image
from FaceSegmentation import FaceSegmentation
import keras.backend as K


def load_image(im, img_size):
    img = image.load_img(im, target_size=(img_size, img_size))
    img = np.expand_dims(img, 0)
    return img

def flatcat(l):
    return tf.concat([tf.reshape(x,[-1]) for x in l],axis=0)

def build_gaussian_mask(n,n_init):
    X=np.linspace(-1,1,n)
    Y=np.linspace(-1,1,n)
    M = [np.exp(-np.power(1*x*x+1.75*y*y,4)) for x in X for y in Y]
    M = np.array(M)

    M = np.reshape(M,(1,n,n,1))

    M=np.tile(M,(n_init,1,1,3))

    return M;

class SROptimizer:
    def __init__(self, G, args, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)

        self.opt_name = args.optimizer
        self.steps = args.steps
        self.min_op = None

        self.img_size = args.img_size
        self.layers = args.layers
        self.n_init = args.n_init
        self.mask_type = args.mask_type

        self.perceptual_model = None

        self.lambda_l1 = None
        self.lambda_l2 = None
        self.lambda_vgg = None
        self.lambda_reg = None
        self.lambda_cross = None

        self.loss_l1 = None
        self.loss_l2 = None
        self.loss_vgg = None
        self.loss_reg = None
        self.loss_cross = None
        self.loss_list = [[],[]]

        self.parse_loss(args.loss)

        self.ref_image_features = None
        self.ref_image = None

        self.latent=G.latent
        self.generated_image=G.generated_image

        self.gauss_mean, self.gauss_scale = np.load("gaussian_fit.npy")

    def apply_mask(self,image):
        if(isinstance(image,np.ndarray)): image = tf.constant(image,dtype='float32')
        resized_mask = tf.image.resize_bicubic(self.image_mask,(image.shape[1],image.shape[2]))[:,:,:,0]
        return tf.einsum('abc,abcd -> abcd',resized_mask,image)

    def build_loss(self):
        self.generated_image = tf.image.resize_bicubic(self.generated_image,(self.img_size, self.img_size))

        self.image_mask = tf.get_variable('image_mask',shape=(self.n_init,self.img_size,self.img_size,3))

        masked_image = self.apply_mask(self.generated_image)

        self.loss = 0

        if(self.lambda_l1 is not None):
            print("\t Building L1")
            self.ref_image = tf.get_variable('ref_img', shape=(self.n_init,self.img_size,self.img_size,3),
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.loss_l1 = 20.0*tf.reduce_mean(tf.abs(self.ref_image/255. - masked_image/255.))

            self.loss += self.lambda_l1 * self.loss_l1
            self.loss_list[1].append(self.loss_l1)
        if(self.lambda_l2 is not None):
            print("\t Building L2")
            self.ref_image = tf.get_variable('ref_img', shape=(self.n_init,self.img_size,self.img_size,3),
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.loss_l2 = 100.0*tf.reduce_mean(tf.square(self.ref_image/255. - masked_image/255.))

            self.loss += self.lambda_l2 * self.loss_l2
            self.loss_list[1].append(self.loss_l2)
        if(self.lambda_vgg is not None):
            print("\t Building VGG")
            self.build_perceptual_loss()

            self.loss += self.lambda_vgg * self.loss_vgg
            self.loss_list[1].append(self.loss_vgg)
        if(self.lambda_reg is not None):
            print("\t Building REG")
            corrected_latent = self.latent - 0.8*tf.nn.relu(self.latent)
            self.loss_reg = tf.reduce_sum(tf.square(corrected_latent-self.gauss_mean)/self.gauss_scale)/512.0

            self.loss += self.lambda_reg * self.loss_reg
            self.loss_list[1].append(self.loss_reg)
        if(self.lambda_cross is not None):
            print("\t Building CROSS")
            A = tf.reshape(tf.transpose(self.latent,(1,2,0)),(18,self.n_init*512))
            r = tf.reduce_sum(A*A, 1)
            r = tf.reshape(r, [-1, 1])
            D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
            self.loss_cross = tf.reduce_sum(D)/2500.0
            self.loss += self.lambda_cross * self.loss_cross
            self.loss_list[1].append(self.loss_cross)

        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[])
        if(self.opt_name=='SGD'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_ph)
        elif(self.opt_name=='ADAM'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        elif(self.opt_name=='SGDM'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_ph,momentum=0.9)
        else:
            raise Exception(f"Invalid optimizer {self.opt_name}")

        self.min_op = optimizer.minimize(self.loss, var_list=[self.latent])
        self.sess.run(tf.variables_initializer(optimizer.variables()))

    def build_perceptual_loss(self):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, [vgg16.layers[i].output for i in self.layers])

        preprocessed_generated_image = preprocess_input(self.generated_image)
        generated_image_features = self.perceptual_model(preprocessed_generated_image)
        generated_image_features = generated_image_features if(isinstance(generated_image_features, list)) else [generated_image_features]
        masked_features = [self.apply_mask(feature) for feature in generated_image_features]
        masked_features=flatcat(masked_features)
        self.ref_image_features = tf.get_variable('ref_img_features', shape=masked_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())

        self.loss_vgg = tf.losses.mean_squared_error(masked_features,self.ref_image_features)/82890.0

    def build_mask(self,image,size):
        loaded_image = load_image(image, size)
        if(self.mask_type is not None):
            if(self.mask_type=="SEGMENT"):
                face_segmentation = FaceSegmentation()
                image_mask = np.tile(np.reshape(face_segmentation.run(Image.fromarray(loaded_image[0])),(1,size,size,1)),(1,1,1,3))
                print("Built mask from image")
            elif(self.mask_type=="PREBUILT"):
                image_mask = load_image("face_mask.png",size)/255
                print("Built mask from prebuilt image")
            elif(self.mask_type=="GAUSSIAN"):
                image_mask = build_gaussian_mask(size,self.n_init)
                print("Built mask from gaussian")
            else:
                raise Exception('Mask type not recognized')
        else:
            image_mask = np.ones(image.shape)
            print("No mask")

        return image_mask
    def set_reference_image(self, image):
        loaded_image = load_image(image, self.img_size)
        image_mask = self.build_mask(image,self.img_size)
        self.sess.run(tf.assign(self.image_mask,image_mask))
        masked_image = self.apply_mask(loaded_image)

        if(self.ref_image_features is not None):
            preprocessed_image = preprocess_input(loaded_image)
            ref_features = self.perceptual_model.predict_on_batch(preprocessed_image)
            ref_features = ref_features if(isinstance(ref_features, list)) else [ref_features]
            masked_ref_features = [self.apply_mask(feature) for feature in ref_features]
            masked_ref_features = flatcat(masked_ref_features)

            self.sess.run(tf.assign(self.ref_image_features, masked_ref_features))

        if(self.ref_image is not None):
            self.sess.run(tf.assign(self.ref_image, masked_image))

    def step(self,lr):
        _, loss_list, loss = self.sess.run([self.min_op, self.loss_list[1], self.loss],feed_dict={self.learning_rate_ph: lr})
        return list(zip(self.loss_list[0],loss_list)), loss

    def parse_loss(self,loss_str):
        for loss_term in loss_str.split('+'):
            weight, loss_type = loss_term.split('*')
            if(weight!=0):
                self.loss_list[0].append(loss_type)
                if(loss_type=="L1"):
                    self.lambda_l1 = float(weight)
                elif(loss_type=="L2"):
                    self.lambda_l2 = float(weight)
                elif(loss_type=="VGG"):
                    self.lambda_vgg = float(weight)
                elif(loss_type=="REG"):
                    self.lambda_reg = float(weight)
                elif(loss_type=="CROSS"):
                    self.lambda_cross = float(weight)
                else:
                    raise Exception(f"Invalid loss type: {loss_type}")

