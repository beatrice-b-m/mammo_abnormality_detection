import tensorflow.keras.backend as k
import tensorflow as tf


class BiSeNetV2(tf.keras.Model):
    def __init__(self, in_shape, num_classes):
        super(BiSeNetV2, self).__init__()

        self.in_shape = in_shape

        # detail branch layers
        self.conv_64_1 = tf.keras.layers.Conv2D(64,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding='same')
        self.conv_64_2 = tf.keras.layers.Conv2D(64,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding='same')
        self.conv_64_3 = tf.keras.layers.Conv2D(64,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding='same')

        self.conv_64_strides2_1 = tf.keras.layers.Conv2D(64,
                                                         kernel_size=(3, 3),
                                                         strides=(2, 2),
                                                         padding='same')
        self.conv_64_strides2_2 = tf.keras.layers.Conv2D(64,
                                                         kernel_size=(3, 3),
                                                         strides=(2, 2),
                                                         padding='same')

        self.conv_128_1 = tf.keras.layers.Conv2D(128,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding='same')
        self.conv_128_2 = tf.keras.layers.Conv2D(128,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding='same')

        self.conv_128_strides2 = tf.keras.layers.Conv2D(128,
                                                        kernel_size=(3, 3),
                                                        strides=(2, 2),
                                                        padding='same')

        # semantic branch layers
        self.stem = SemanticStem(16)
        self.ge_32 = GatherExpansionStride1(32)
        self.ge_32_strides2 = GatherExpansionStride2(32)
        self.ge_64 = GatherExpansionStride1(64)
        self.ge_64_strides2 = GatherExpansionStride2(64)
        self.ge_128_1 = GatherExpansionStride1(128)
        self.ge_128_2 = GatherExpansionStride1(128)
        self.ge_128_3 = GatherExpansionStride1(128)
        self.ge_128_strides2 = GatherExpansionStride2(128)
        self.ce = ContextEmbedding(128)

        # agg layers
        self.agg = BilateralAggregation(128)

        # booster layers
        self.seg_head = SegmentationHead(num_classes)

    def call(self, inputs, training=False):
        # detail branch stage 1
        detail_1 = self.conv_64_strides2_1(inputs)
        detail_1 = self.conv_64_1(detail_1)

        # semantic branch stage 1
        semantic_1 = self.stem(inputs)

        # detail branch stage 2
        detail_2 = self.conv_64_strides2_2(detail_1)
        detail_2 = self.conv_64_2(detail_2)
        detail_2 = self.conv_64_3(detail_2)

        # detail branch stage 3
        detail_3 = self.conv_128_strides2(detail_2)
        detail_3 = self.conv_128_1(detail_3)
        detail_3 = self.conv_128_2(detail_3)

        # semantic branch stage 3
        semantic_3 = self.ge_32_strides2(semantic_1)
        semantic_3 = self.ge_32(semantic_3)

        # semantic branch stage 4
        semantic_4 = self.ge_64_strides2(semantic_3)
        semantic_4 = self.ge_64(semantic_4)

        # semantic branch stage 5
        semantic_5 = self.ge_128_strides2(semantic_4)
        semantic_5 = self.ge_128_1(semantic_5)
        semantic_5 = self.ge_128_2(semantic_5)
        semantic_5 = self.ge_128_3(semantic_5)
        #         print(f'{semantic_5.shape = }')
        semantic_5 = self.ce(semantic_5)

        # aggregation layer and output segmentation head
        agg_out = self.agg(detail_3, semantic_5)
        out = self.seg_head(agg_out)
        return out

    def build_graph(self):
        x = tf.keras.layers.Input(self.in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class SemanticStem(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num,
                 padding: str = 'same',
                 c_axis=-1):
        super(SemanticStem, self).__init__()

        self.conv_3x3 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding=padding)

        self.conv_3x3_stride2_1 = tf.keras.layers.Conv2D(filter_num,
                                                         kernel_size=(3, 3),
                                                         strides=(2, 2),
                                                         padding=padding)

        self.conv_3x3_stride2_2 = tf.keras.layers.Conv2D(filter_num,
                                                         kernel_size=(3, 3),
                                                         strides=(2, 2),
                                                         padding=padding)

        self.conv_1x1 = tf.keras.layers.Conv2D(filter_num // 2,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding=padding)

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                  strides=(2, 2),
                                                  padding=padding)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()

        self.concat = tf.keras.layers.Concatenate(axis=c_axis)

    def call(self, inputs, training=False):
        # input path
        x = self.conv_3x3_stride2_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        # path 1
        x1 = self.conv_1x1(x)
        x1 = self.bn_2(x1, training=training)
        x1 = tf.nn.relu(x1)

        x1 = self.conv_3x3_stride2_2(x1)
        x1 = self.bn_3(x1, training=training)
        x1 = tf.nn.relu(x1)

        # path 2
        x2 = self.max_pool(x)

        # combined path
        x3 = self.concat([x1, x2])
        x3 = self.conv_3x3(x3)
        x3 = self.bn_4(x3, training=training)
        x3 = tf.nn.relu(x3)
        return x3

    def build_graph(self, in_shape):
        x = tf.keras.layers.Input(in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class GatherExpansionStride1(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num: int,
                 padding: str = 'same',
                 depth_multiplier: int = 6):
        super(GatherExpansionStride1, self).__init__()

        self.conv_3x3 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding=padding)

        self.conv_1x1 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding=padding)

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding=padding,
                                                       depth_multiplier=depth_multiplier)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # path 1
        x = self.conv_3x3(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.dw_conv(x)
        x = self.bn_2(x, training=training)

        x = self.conv_1x1(x)
        x = self.bn_3(x, training=training)

        # combined path
        x = tf.keras.layers.Add()([x, inputs])
        x = tf.nn.relu(x)
        return x

    def build_graph(self, in_shape):
        x = tf.keras.layers.Input(in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class GatherExpansionStride2(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num: int,
                 padding: str = 'same',
                 depth_multiplier: int = 6):
        super(GatherExpansionStride2, self).__init__()

        self.conv_3x3 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding=padding)

        self.conv_1x1_1 = tf.keras.layers.Conv2D(filter_num,
                                                 kernel_size=(1, 1),
                                                 strides=(1, 1),
                                                 padding=padding)

        self.conv_1x1_2 = tf.keras.layers.Conv2D(filter_num,
                                                 kernel_size=(1, 1),
                                                 strides=(1, 1),
                                                 padding=padding)

        self.dw_conv_stride2_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                                 strides=(2, 2),
                                                                 padding=padding,
                                                                 depth_multiplier=depth_multiplier)

        self.dw_conv_stride2_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                                 strides=(2, 2),
                                                                 padding=padding,
                                                                 depth_multiplier=depth_multiplier)

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding=padding,
                                                       depth_multiplier=depth_multiplier // 6)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.bn_6 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # path 1
        x1 = self.conv_3x3(inputs)
        x1 = self.bn_1(x1, training=training)
        x1 = tf.nn.relu(x1)

        x1 = self.dw_conv_stride2_1(x1)
        x1 = self.bn_2(x1, training=training)

        x1 = self.dw_conv(x1)
        x1 = self.bn_3(x1, training=training)

        x1 = self.conv_1x1_1(x1)
        x1 = self.bn_4(x1, training=training)

        # path 2
        x2 = self.dw_conv_stride2_2(inputs)
        x2 = self.bn_5(x2, training=training)

        x2 = self.conv_1x1_2(x2)
        x2 = self.bn_6(x2, training=training)

        # combined path
        x = tf.keras.layers.Add()([x1, x2])
        x = tf.nn.relu(x)
        return x

    def build_graph(self, in_shape):
        x = tf.keras.layers.Input(in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class ContextEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num,
                 padding: str = 'same'):
        super(ContextEmbedding, self).__init__()
        self.filter_num = filter_num

        self.ga_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.conv_3x3 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding=padding)

        self.conv_1x1 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding=padding)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # path 1
        x = self.ga_pool(inputs)
        x = tf.keras.layers.Reshape((1, 1, -1))(x)  # reshape from (None, c) to (None, 1, 1, c)
        x = self.bn_1(x, training=training)

        x = self.conv_1x1(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = tf.broadcast_to(x, [k.shape(inputs)[0], 32, 26, self.filter_num])

        # combined path
        x = tf.keras.layers.Add()([x, inputs])
        x = self.conv_3x3(x)
        return x

    def build_graph(self, in_shape):
        x = tf.keras.layers.Input(in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model


class BilateralAggregation(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num,
                 padding: str = 'same',
                 depth_multiplier: int = 1):
        super(BilateralAggregation, self).__init__()

        self.dw_conv_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                         strides=(1, 1),
                                                         padding=padding,
                                                         depth_multiplier=depth_multiplier)

        self.dw_conv_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                         strides=(1, 1),
                                                         padding=padding,
                                                         depth_multiplier=depth_multiplier)

        self.conv_1x1_1 = tf.keras.layers.Conv2D(filter_num,
                                                 kernel_size=(1, 1),
                                                 strides=(1, 1),
                                                 padding=padding)

        self.conv_1x1_2 = tf.keras.layers.Conv2D(filter_num,
                                                 kernel_size=(1, 1),
                                                 strides=(1, 1),
                                                 padding=padding)

        self.conv_3x3_1 = tf.keras.layers.Conv2D(filter_num,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding=padding)

        self.conv_3x3_2 = tf.keras.layers.Conv2D(filter_num,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding=padding)

        self.conv_3x3_stride2 = tf.keras.layers.Conv2D(filter_num,
                                                       kernel_size=(3, 3),
                                                       strides=(2, 2),
                                                       padding=padding)

        self.avg_pool_3x3_stride2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                     strides=(2, 2),
                                                                     padding=padding)

        self.upsample_4x4_1 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upsample_4x4_2 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.bn_5 = tf.keras.layers.BatchNormalization()

    def call(self, detail_in, semantic_in, training=False):
        # path 1, input = detail_in
        x1 = self.dw_conv_1(detail_in)  # dw conv
        x1 = self.bn_1(x1, training=training)  # bn
        x1 = self.conv_1x1_1(x1)  # 1x1 conv

        # path 2, input = detail_in
        x2 = self.conv_3x3_stride2(detail_in)  # 3x3 conv stride 2
        x2 = self.bn_2(x2, training=training)  # bn
        x2 = self.avg_pool_3x3_stride2(x2)  # 3x3 average pooling stride 2

        # path 3, input = semantic_in
        x3 = self.conv_3x3_1(semantic_in)  # 3x3 conv
        x3 = self.bn_3(x3, training=training)  # bn
        x3 = self.upsample_4x4_1(x3)  # 4x4 upsample
        x3 = tf.nn.sigmoid(x3)  # sigmoid

        # path 4, input = semantic_in
        x4 = self.dw_conv_2(semantic_in)  # 3x3 dw conv
        x4 = self.bn_4(x4, training=training)  # bn
        x4 = self.conv_1x1_2(x4)  # 1x1 conv
        x4 = tf.nn.sigmoid(x4)  # sigmoid

        # path L, combine x1 & x3
        x5 = tf.keras.layers.Multiply()([x1, x3])

        # path R, combine x2 & x4
        x6 = tf.keras.layers.Multiply()([x2, x4])
        x6 = self.upsample_4x4_2(x6)  # 4x4 upsample

        # output path, combine x5 & x6
        x7 = tf.keras.layers.Add()([x5, x6])
        x7 = self.conv_3x3_2(x7)
        x7 = self.bn_5(x7, training=training)
        return x7

    def build_graph(self, detail_shape, semantic_shape):
        detail_in = tf.keras.layers.Input(detail_shape)
        semantic_in = tf.keras.layers.Input(semantic_shape)
        model = tf.keras.Model(inputs=[detail_in, semantic_in], outputs=self.call(detail_in, semantic_in))
        return model


class SegmentationHead(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num,
                 upsample_size=(8, 8),
                 padding: str = 'same'):
        super(SegmentationHead, self).__init__()

        self.conv_3x3 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding=padding)

        self.conv_1x1 = tf.keras.layers.Conv2D(filter_num,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding=padding)

        self.upsample = tf.keras.layers.UpSampling2D(size=upsample_size, interpolation='bilinear')

        self.bn = tf.keras.layers.BatchNormalization()
        # ADD UPSAMPLE IF NECESSARY

    def call(self, inputs, training=False):
        # path 1
        x = self.conv_3x3(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_1x1(x)
        x = self.upsample(x)
        x = tf.nn.sigmoid(x)
        return x

    def build_graph(self, in_shape):
        x = tf.keras.layers.Input(in_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model
