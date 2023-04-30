import tensorflow as tf


class ResNet50V2(tf.keras.Model):
    def __init__(self, in_shape):
        super(ResNet50V2, self).__init__()

        self.in_shape = in_shape

        self.conv_7x7_s2 = tf.keras.layers.Conv2D(64,
                                                  kernel_size=(7, 7),
                                                  strides=(2, 2),
                                                  padding='same',
                                                  kernel_initializer='he_normal')

        self.max_pool_s2 = tf.keras.layers.MaxPool2D(strides=(2, 2), padding='same')  # WHAT PARAMS?
        self.conv_1x1_s1 = tf.keras.layers.Conv2D(256,
                                                  kernel_size=(1, 1),
                                                  strides=(1, 1),
                                                  padding='same',
                                                  kernel_initializer='he_normal')
        self.id_1 = IdentityBlock(64)
        self.id_2 = IdentityBlock(64)
        self.id_3 = IdentityBlock(64)

        self.res_block_2 = ResidualBlock(128, 4)
        self.res_block_3 = ResidualBlock(256, 6)
        self.res_block_4 = ResidualBlock(512, 3)

    def call(self, inputs, training=False):
        # return model output
        x = self.conv_7x7_s2(inputs)
        
        # residual block 1
        x = self.max_pool_s2(x)
        x = self.conv_1x1_s1(x)
        x = self.id_1(x)
        x = self.id_2(x)
        x = self.id_3(x)
        

        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)

        return x

    def build_graph(self):
        # compile and return model
        x = tf.keras.layers.Input(self.in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, n):
        super(ResidualBlock, self).__init__()
        # get convolution block to reduce dims by half
        self.conv_block = ConvolutionBlock(filter_num)

        # get list of identity blocks
        self.id_block_list = []
        for i in range(n - 1):
            self.id_block_list.append(IdentityBlock(filter_num))

    def call(self, inputs, training=False):
        # return model output
        x = self.conv_block(inputs)

        for id_block in self.id_block_list:
            x = id_block(x)

        return x

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model


# reduces dims by half
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, kernel_init: str = 'he_normal'):
        super(ConvolutionBlock, self).__init__()
        self.conv_1x1_s2 = tf.keras.layers.Conv2D(filter_num,
                                                  kernel_size=(1, 1),
                                                  strides=(2, 2),
                                                  padding='same',
                                                  kernel_initializer=kernel_init)
        self.conv_1x1_s1 = tf.keras.layers.Conv2D(4 * filter_num,
                                                  kernel_size=(1, 1),
                                                  strides=(1, 1),
                                                  padding='same',
                                                  kernel_initializer=kernel_init)
        self.conv_1x1_s2_sc = tf.keras.layers.Conv2D(4 * filter_num,
                                                     kernel_size=(1, 1),
                                                     strides=(2, 2),
                                                     padding='same',
                                                     kernel_initializer=kernel_init)
        self.conv_3x3_s1 = tf.keras.layers.Conv2D(filter_num,
                                                  kernel_size=(3, 3),
                                                  strides=(1, 1),
                                                  padding='same',
                                                  kernel_initializer=kernel_init)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # main path
        x = self.bn_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1x1_s2(x)

        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_3x3_s1(x)

        x = self.bn_3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1x1_s1(x)

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
    def __init__(self, filter_num, kernel_init: str = 'he_normal'):
        super(IdentityBlock, self).__init__()
        self.conv_1x1_s1_1 = tf.keras.layers.Conv2D(filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=kernel_init)
        self.conv_1x1_s1_2 = tf.keras.layers.Conv2D(4 * filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    kernel_initializer=kernel_init)
        self.conv_3x3_s1 = tf.keras.layers.Conv2D(filter_num,
                                                  kernel_size=(3, 3),
                                                  strides=(1, 1),
                                                  padding='same',
                                                  kernel_initializer=kernel_init)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # main path
        x = self.bn_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1x1_s1_1(x)

        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_3x3_s1(x)

        x = self.bn_3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1x1_s1_2(x)

        # combined path
        x_out = tf.keras.layers.Add()([x, inputs])
        return x_out

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model
