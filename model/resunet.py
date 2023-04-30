from tensorflow.keras.layers import Input, concatenate, Conv2D, Add, MaxPooling2D, Activation, Dense, Reshape, \
    GlobalAveragePooling2D, Multiply, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.models import Model


# file for original resunet
# function get_res_unet takes in the image rows, columns and the dropout rate as the input.
def get_res_unet(img_rows, img_cols, dropout_rate=0.4):
    # inputs represents an input layer that uses the image rows and columns
    # from the function input along with the number of channels.
    inputs = Input((img_rows, img_cols, 1))

    # Conv1 to Conv4 and Pool1 to Pool4 represents the contracting path.
    # Each contracting path has four convolutional layers with batch
    # normalization, a residual block that uses the residual_block
    # function, max pooling, and dropout rate.
    # Contracting path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = residual_block(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = residual_block(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = residual_block(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = residual_block(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # The bridge layer is created with a convolutional layer, followed by batch normalization, a residual block.
    # Bridge
    bridge = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    bridge = BatchNormalization()(bridge)
    bridge = residual_block(bridge, 512)

    # expansive path has four transposed convolutional layers
    # each path concatenates the corresponding contracting path layer,
    # two convolutional layers with batch normalization, and residual block.
    # Expansive path
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bridge)
    up1 = concatenate([up1, conv4], axis=3)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    conv5 = BatchNormalization()(conv5)
    conv5 = residual_block(conv5, 256)

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up2 = concatenate([up2, conv3], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    conv6 = BatchNormalization()(conv6)
    conv6 = residual_block(conv6, 128)

    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up3 = concatenate([up3, conv2], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    conv7 = BatchNormalization()(conv7)
    conv7 = residual_block(conv7, 64)

    up4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up4 = concatenate([up4, conv1], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up4)
    conv8 = BatchNormalization()(conv8)
    conv8 = residual_block(conv8, 32)

    # the output layer creates a single convolutional layer with sigmoid activation.
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    # The model is then created with the input and output layers.
    model = Model(inputs=[inputs], outputs=[outputs])

    # returns the model
    return model


# The residual block function takes in an input layer and number of filters.
# It applies two convolutional layers with batch normalization.
def residual_block(inputs, filters):
    # Residual block with two 3x3 convolutions
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    # Adds the output to the input, applies a ReLU activation, and returns the output.
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x
