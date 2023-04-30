# code for 'Separable ResNet50v2?' name pending also maybe someone has already done this???

import os
import tensorflow.keras.backend as k
import tensorflow as tf


class SepResNet101V2(tf.keras.Model):
    def __init__(self, in_shape, num_classes):
        super(SepResNet101V2, self).__init__()

        self.in_shape = in_shape

        # since the input has only 1 channel, this layer has very few parameters
        # even as a non-separable convolution
        self.conv_7x7_s2 = tf.keras.layers.Conv2D(32,
                                                  kernel_size=(7, 7),
                                                  dilation_rate=(1, 1),
                                                  strides=(2, 2),
                                                  padding='same',
                                                  kernel_initializer='he_normal')

        #         self.max_pool_s2 = tf.keras.layers.MaxPool2D(strides=(2, 2), padding='same')  # WHAT PARAMS?
        self.sep_conv_3x3_s2 = tf.keras.layers.SeparableConv2D(64,
                                                               kernel_size=(3, 3),
                                                               strides=(2, 2),
                                                               padding='same',
                                                               dilation_rate=(1, 1),
                                                               depth_multiplier=1,
                                                               depthwise_initializer='he_normal',
                                                               pointwise_initializer='he_normal')

        self.res_block_1 = ResidualBlock(64, 3)
        self.res_block_2 = ResidualBlock(128, 4, strides=2)
        self.res_block_3 = ResidualBlock(256, 23, strides=2)
        self.res_block_4 = ResidualBlock(256, 3, dilation_rate=(2, 2))

        self.clf = Classifier(num_classes)

    def call(self, inputs, training=False):
        # return model output
        x = self.conv_7x7_s2(inputs)

        # residual block 1
        x = self.sep_conv_3x3_s2(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)

        # classifier for pretraining
        x = self.clf(x)
        return x

    def build_graph(self):
        # compile and return model
        x = tf.keras.layers.Input(self.in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, n, strides=1, dilation_rate=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.block_list = []
        if strides == 2:
            # get convolution block to reduce dims by half
            self.block_list.append(ConvolutionBlock(filter_num, strides=(strides, strides)))

            # get list of identity blocks
            for i in range(n - 1):
                self.block_list.append(IdentityBlock(filter_num, dilation_rate))

        else:
            # get list of identity blocks
            for i in range(n):
                self.block_list.append(IdentityBlock(filter_num, dilation_rate))

    def call(self, inputs, training=False):
        # return model output
        # x = self.conv_block(inputs)
        x = inputs
        for block in self.block_list:
            x = block(x)

        return x

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model


# reduces dims by half
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, strides, kernel_init: str = 'he_normal', depth_mult: int = 1):
        super(ConvolutionBlock, self).__init__()
        self.conv_1x1_s2 = tf.keras.layers.Conv2D(filter_num // 2,
                                                  kernel_size=(1, 1),
                                                  strides=strides,
                                                  padding='same',
                                                  kernel_initializer=kernel_init)
        self.conv_1x1_s2_sc = tf.keras.layers.Conv2D(filter_num,
                                                     kernel_size=(1, 1),
                                                     strides=strides,
                                                     padding='same',
                                                     kernel_initializer=kernel_init)
        self.sep_conv_3x3_s1 = tf.keras.layers.SeparableConv2D(filter_num,
                                                               kernel_size=(3, 3),
                                                               strides=(1, 1),
                                                               padding='same',
                                                               depth_multiplier=depth_mult,
                                                               depthwise_initializer=kernel_init,
                                                               pointwise_initializer=kernel_init)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        # self.bn_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # main path
        x = self.bn_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1x1_s2(x)

        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.sep_conv_3x3_s1(x)

        # shortcut
        sc = self.conv_1x1_s2_sc(inputs)

        # combined path
        x_out = tf.keras.layers.Add()([x, sc])
        return x_out

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, dilation_rate, kernel_init: str = 'he_normal', depth_mult: int = 1):
        super(IdentityBlock, self).__init__()
        self.conv_1x1_s1_1 = tf.keras.layers.Conv2D(filter_num // 2,
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=kernel_init)
        self.sep_conv_3x3_s1 = tf.keras.layers.SeparableConv2D(filter_num,
                                                               kernel_size=(3, 3),
                                                               strides=(1, 1),
                                                               padding='same',
                                                               dilation_rate=dilation_rate,
                                                               depth_multiplier=depth_mult,
                                                               depthwise_initializer=kernel_init,
                                                               pointwise_initializer=kernel_init)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # main path
        x = self.bn_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1x1_s1_1(x)

        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.sep_conv_3x3_s1(x)

        # combined path
        x_out = tf.keras.layers.Add()([x, inputs])
        return x_out

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model


class Classifier(tf.keras.layers.Layer):
    def __init__(self, num_classes, kernel_init: str = 'he_normal'):
        super(Classifier, self).__init__()
        #         self.avg_pool = tf.keras.layers.AveragePooling2D(padding='same')
        self.sep_conv_3x3_s2_1 = tf.keras.layers.SeparableConv2D(512,
                                                                 kernel_size=(3, 3),
                                                                 strides=(2, 2),
                                                                 padding='same',
                                                                 dilation_rate=(1, 1),
                                                                 depth_multiplier=1,
                                                                 depthwise_initializer='he_normal',
                                                                 pointwise_initializer='he_normal')
        self.sep_conv_3x3_s2_2 = tf.keras.layers.SeparableConv2D(128,
                                                                 kernel_size=(3, 3),
                                                                 strides=(2, 2),
                                                                 padding='same',
                                                                 dilation_rate=(1, 1),
                                                                 depth_multiplier=1,
                                                                 depthwise_initializer='he_normal',
                                                                 pointwise_initializer='he_normal')
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(32, kernel_initializer=kernel_init)
        self.dense_out = tf.keras.layers.Dense(num_classes, kernel_initializer=kernel_init)

    def call(self, inputs):
        # reduce dimensions
        #         x = self.avg_pool(inputs)
        x = self.sep_conv_3x3_s2_1(inputs)
        x = self.sep_conv_3x3_s2_2(x)

        # flatten output
        x = self.flat(x)

        # fully connected layers with sigmoid activation
        x = self.dense_1(x)
        x = self.dense_out(x)
        x = tf.nn.sigmoid(x)
        return x

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model
