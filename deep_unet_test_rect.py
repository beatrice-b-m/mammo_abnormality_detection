import model.deep_unet_decoder as dec
from model.encoder import sepresnet152v2_c
from utility.loss import seg_loss, dice_coef, iou_coef
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import binarize
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def npy_loader(maindir, seed: int = 2):
    # Get list of files in directory
    directory = maindir + '*.npy'
    pathlist = glob.glob(directory)
    
    # Iterate over list of files
    for path in pathlist:
        array = np.load(path)
        img = array[:, :, 0]
        img = (img * 2) - 1
        mask = binarize(array[:, :, 1])
        yield img[..., np.newaxis], mask[..., np.newaxis]


def npy_dataset(maindir, shape_i, shape_m, seed: int = 2, batch: int = 1):
    ds = tf.data.Dataset.from_generator(lambda: npy_loader(maindir=maindir, seed = seed), 
                                        output_types=(tf.float32, tf.float32), 
                                        output_shapes=(shape_i, shape_m))
    return ds.batch(batch)


batch_size = 16

train_df = npy_dataset('/home/careinfolab/unet_mammo/images/pos_norm/train/', 
                       (1024, 832, 1),
                       (1024, 832, 1),
                       seed=2, 
                       batch=batch_size)

val_df = npy_dataset('/home/careinfolab/unet_mammo/images/pos_norm/val/', 
                       (1024, 832, 1),
                       (1024, 832, 1),
                       seed=2, 
                       batch=batch_size)

test_df = npy_dataset('/home/careinfolab/unet_mammo/images/pos_norm/test/', 
                       (1024, 832, 1),
                       (1024, 832, 1),
                       seed=2, 
                       batch=batch_size)

enc = sepresnet152v2_c.SepResNet152V2((1024, 832, 1), 1)
encoder = enc.build_graph()
encoder.load_weights('./saved_models/sepresnet152v2_enc_c2')

# encoder.trainable = False

# encoder.summary()
decoder = dec.DeepUNet()
model = decoder.build_graph(encoder.input, 
                            encoder.layers[-2].output, 
                            encoder.layers[3].output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=[seg_loss], metrics=[dice_coef, iou_coef])
name = './saved_models/deep_unet_test152_rect'

model_checkpoint = ModelCheckpoint(name, monitor='val_loss', save_best_only=True)

history = model.fit(train_df, epochs=150, verbose=1, shuffle=True,
                    validation_data=val_df,
                    callbacks=[model_checkpoint])
