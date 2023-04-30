from model import resunet
from utility.loss import seg_loss, dice_coef, iou_coef
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
    ds = tf.data.Dataset.from_generator(lambda: npy_loader(maindir=maindir, seed=seed),
                                        output_types=(tf.float16, tf.float16),
                                        output_shapes=(shape_i, shape_m))
    return ds.batch(batch)


# load data
batch_size = 4

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

# define model
model = resunet.get_res_unet(1024, 832)
model.compile(optimizer=Adam(), loss=[seg_loss], metrics=[dice_coef, iou_coef])
name = './saved_models/resunet_test1_rect'

# get callback functions
model_checkpoint = ModelCheckpoint(name, monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)

history = model.fit(train_df, epochs=150, verbose=1, shuffle=True,
                    validation_data=val_df,
                    callbacks=[model_checkpoint, early_stopping])
