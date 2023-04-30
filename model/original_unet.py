import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Dropout
import atexit


# create a UNetBase object
class UNetBase:
    # constructor method that takes learning rate, input shape, architecture type and dropout rate as inputs
    def __init__(self, lr, input_shape, architecture: str = 'base', drop: float = 0.5):
        # assign input parameters to class variables
        self.learning_rate = lr
        self.architecture = architecture
        self.input_shape = input_shape
        self.drop = drop
        self.model = None

    # method to compile the model based on the architecture type
    def compile_model(self):
        # check if architecture is 'base'
        if self.architecture == 'base':
            # call unet_original method to create the model
            self.model = self.unet_original()
            # return the compiled model
            return self.model

    # method to create the UNet model architecture
    def unet_original(self):
        # distribute strategy for multi-GPU training
        mirrored_strategy = tf.distribute.MirroredStrategy()

        # register the close method of the collective operation pool to be executed at the exit of the program
        atexit.register(mirrored_strategy._extended._collective_ops._pool.close)

        # use the distributed strategy for creating the model
        with mirrored_strategy.scope():
            # input
            inputs = Input(self.input_shape)
            # conv block 1, 64, connect c1 to block 9
            c1 = conv2d_layer(inputs, 64)
            c1 = conv2d_layer(c1, 64)
            p1 = MaxPooling2D(pool_size=(2,2))(c1)

            # conv block 2, 128, connect c2 to block 8
            c2 = conv2d_layer(p1, 128)
            c2 = conv2d_layer(c2, 128)
            p2 = MaxPooling2D(pool_size=(2,2))(c2)

            # conv block 3, 256, connect c3 to block 7
            c3 = conv2d_layer(p2, 256)
            c3 = conv2d_layer(c3, 256)
            p3 = MaxPooling2D(pool_size=(2,2))(c3)

            # conv block 4 with dropout, 512, connect d4 to block 6
            c4 = conv2d_layer(p3, 512)
            c4 = conv2d_layer(c4, 512)
            d4 = Dropout(self.drop)(c4)  # not found in original paper
            p4 = MaxPooling2D(pool_size=(2,2))(c4)

            # conv block 5 with dropout, 1024
            c5 = conv2d_layer(p4, 1024)
            c5 = conv2d_layer(c5, 1024)
            d5 = Dropout(self.drop)(c5)  # not found in original paper

            # start upsampling
            # conv block 6, 512
            u6 = conv2d_layer(d5, 512, kernel_size=2, upsample=True)
            m6 = concatenate([d4, u6], axis=3)
            c6 = conv2d_layer(m6, 512)
            c6 = conv2d_layer(c6, 512)

            # conv block 7, 256
            u7 = conv2d_layer(c6, 256, kernel_size=2, upsample=True)
            m7 = concatenate([c3, u7], axis=3)
            c7 = conv2d_layer(m7, 256)
            c7 = conv2d_layer(c7, 256)

            # conv block 8, 128
            u8 = conv2d_layer(c7, 128, kernel_size=2, upsample=True)
            m8 = concatenate([c2, u8], axis=3)
            c8 = conv2d_layer(m8, 128)
            c8 = conv2d_layer(c8, 128)

            # conv block 9, 64
            u9 = conv2d_layer(c8, 64, kernel_size=2, upsample=True)
            m9 = concatenate([c1, u9], axis=3)
            c9 = conv2d_layer(m9, 64)
            c9 = conv2d_layer(c9, 64)

            # conv block 10, output
            c10 = conv2d_layer(c9, 2)
            outputs = conv2d_layer(c10, 1, act='sigmoid', pad='same', kernel_init='glorot_uniform')

            # the model is then created with the input and output layers.
            model = Model(inputs=inputs, outputs=outputs)
        return model


# conv2d_layer function takes input of input layers, filters in the layer, kernel size
def conv2d_layer(in_layer, filters, kernel_size: int = 3, act: str = 'relu', pad: str = 'same',
                 kernel_init: str = 'he_normal', upsample: bool = False):
    # if not an upsampling layer
    if not upsample:
        # creates a convolutional layer
        conv_layer = Conv2D(filters,
                            kernel_size,
                            activation=act,
                            padding=pad,
                            kernel_initializer=kernel_init)(in_layer)
    else:
        # conv_layer represents an upsampling layer which upsamples the
        # input layer before passing it through a convolutional layer.
        conv_layer = Conv2D(filters,
                            kernel_size,
                            activation=act,
                            padding=pad,
                            kernel_initializer=kernel_init)(UpSampling2D(size=(2,2))(in_layer))
    return conv_layer
