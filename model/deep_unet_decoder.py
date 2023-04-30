import tensorflow as tf


class DeepUNet(tf.keras.Model):
    def __init__(self, num_classes=1, padding='same', kernel_init='he_normal'):
        super(DeepUNet, self).__init__()
        self.aspp = AtrousSpatialPyramidPooling(256)

        self.upsample_1 = tf.keras.layers.UpSampling2D(size=(4, 4),
                                                       interpolation='bilinear')
        self.upsample_2 = tf.keras.layers.UpSampling2D(size=(4, 4),
                                                       interpolation='bilinear')

        self.conv_1x1 = tf.keras.layers.Conv2D(32,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding=padding,
                                               kernel_initializer=kernel_init)
        self.dw_conv_3x3 = tf.keras.layers.SeparableConv2D(num_classes,
                                                           kernel_size=(3, 3),
                                                           strides=(1, 1),
                                                           padding=padding,
                                                           dilation_rate=(1, 1),
                                                           depth_multiplier=1,
                                                           depthwise_initializer=kernel_init,
                                                           pointwise_initializer=kernel_init)

    def call(self, x_out, x_low):
        # get encoder outputs
        # x_out, x_low = self.encoder(inputs)

        # aspp path
        aspp_out = self.aspp(x_out)
        aspp_out = self.upsample_1(aspp_out)

        # low level path
        x_low = self.conv_1x1(x_low)

        # combine
        x = tf.keras.layers.Concatenate()([x_low, aspp_out])
        x = self.dw_conv_3x3(x)
        x = self.upsample_2(x)
        x = tf.nn.sigmoid(x)
        return x

    def build_graph(self, encoder_input, encoder_out, encoder_low):
        # compile and return model
        model = tf.keras.Model(inputs=encoder_input, outputs=self.call(encoder_out, encoder_low))
        return model


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, num_filters, padding='same', kernel_init='he_normal', out_channels=32):
        super(AtrousSpatialPyramidPooling, self).__init__()

        self.conv_1x1 = tf.keras.layers.Conv2D(num_filters,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding=padding,
                                               kernel_initializer=kernel_init)
        self.conv_1x1_out = tf.keras.layers.Conv2D(out_channels,
                                                   kernel_size=(1, 1),
                                                   strides=(1, 1),
                                                   padding=padding,
                                                   kernel_initializer=kernel_init)
        self.atrous_conv_3x3_r2 = tf.keras.layers.SeparableConv2D(num_filters,
                                                                  kernel_size=(3, 3),
                                                                  strides=(1, 1),
                                                                  padding=padding,
                                                                  dilation_rate=(2, 2),
                                                                  depth_multiplier=1,
                                                                  depthwise_initializer=kernel_init,
                                                                  pointwise_initializer=kernel_init)
        self.atrous_conv_3x3_r4 = tf.keras.layers.SeparableConv2D(num_filters,
                                                                  kernel_size=(3, 3),
                                                                  strides=(1, 1),
                                                                  padding=padding,
                                                                  dilation_rate=(4, 4),
                                                                  depth_multiplier=1,
                                                                  depthwise_initializer=kernel_init,
                                                                  pointwise_initializer=kernel_init)
        self.atrous_conv_3x3_r6 = tf.keras.layers.SeparableConv2D(num_filters,
                                                                  kernel_size=(3, 3),
                                                                  strides=(1, 1),
                                                                  padding=padding,
                                                                  dilation_rate=(6, 6),
                                                                  depth_multiplier=1,
                                                                  depthwise_initializer=kernel_init,
                                                                  pointwise_initializer=kernel_init)
        self.glob_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.upsample = tf.keras.layers.UpSampling2D(size=(64, 52),
                                                     interpolation='bilinear')

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.bn_6 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # branch 1, 1x1 conv
        x1 = self.conv_1x1(inputs)
        x1 = self.bn_1(x1, training=training)
        x1 = tf.nn.relu(x1)

        # branch 2, dilation 2
        x2 = self.atrous_conv_3x3_r2(inputs)
        x2 = self.bn_2(x2, training=training)
        x2 = tf.nn.relu(x2)

        # branch 3, dilation 4
        x3 = self.atrous_conv_3x3_r4(inputs)
        x3 = self.bn_3(x3, training=training)
        x3 = tf.nn.relu(x3)

        # branch 4, dilation 6
        x4 = self.atrous_conv_3x3_r6(inputs)
        x4 = self.bn_4(x4, training=training)
        x4 = tf.nn.relu(x4)

        # branch 5, global average pooling
        x5 = self.glob_avg_pool(inputs)  # shape (None, h, w, c) > (None, c)
        x5 = tf.keras.layers.Reshape((1, 1, -1))(x5)  # shape (None, c) > (None, 1, 1, c)
        x5 = self.bn_5(x5, training=training)
        x5 = tf.nn.relu(x5)
        x5 = self.upsample(x5)  # shape (None, 1, 1, c) > (None, 64, 52, c)

        # combine
        x_out = tf.keras.layers.Concatenate()([x1, x2, x3, x4, x5])
        x_out = self.conv_1x1_out(x_out)
        x_out = self.bn_6(x_out)
        x_out = tf.nn.relu(x_out)
        return x_out

    def compile_layers(self, input_shape):
        # compile layers into temp model to test
        x = tf.keras.layers.Input(input_shape)
        temp_model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return temp_model
