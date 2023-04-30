from model.encoder import sepresnet152v2_c
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import atexit
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def preprocess(images, labels):
    # convert images to single channel
    images = tf.image.rgb_to_grayscale(images)

    # normalize images between -1 and 1
    return tf.keras.applications.resnet_v2.preprocess_input(images), labels

def loaddataset(path, imsize, batchsize):
    # retrieve dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=imsize, batch_size=batchsize)

    # preprocess dataset
    dataset = dataset.map(preprocess)
    dataset = dataset.prefetch(buffer_size=3)
    return dataset

base_path = '/media/careinfolab/CI_Lab/unet'
train_dir = '/train'
val_dir = '/val'
test_dir = '/test'

img_shape = (1024, 832)
batchsize = 16
train_df = loaddataset(base_path + train_dir, img_shape, batchsize)
val_df = loaddataset(base_path + val_dir, img_shape, batchsize)
test_df = loaddataset(base_path + test_dir, img_shape, batchsize)

mirrored_strategy = tf.distribute.MirroredStrategy()

# ensures memory is cleared correctly
atexit.register(mirrored_strategy._extended._collective_ops._pool.close)

# define model with distribution strategy
with mirrored_strategy.scope():
    seprnv2 = sepresnet152v2_c.SepResNet152V2((1024, 832, 1), 1)
    model = seprnv2.build_graph()

    metr = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'), 
            tf.keras.metrics.Precision(name='prec'), 
            tf.keras.metrics.Recall(name='rec')]

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='BinaryCrossentropy', metrics=metr)

name = './saved_models/sepresnet152v2_enc_c2'

model_checkpoint = ModelCheckpoint(name, monitor='val_loss', save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=8)


history = model.fit(train_df, epochs=150, verbose=1, shuffle=True,
                    validation_data=val_df,
                    callbacks=[model_checkpoint, early_stopping])
