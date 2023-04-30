from model import bisenet
from utility.loss import seg_loss, dice_coef, iou_coef
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
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
        mask = array[:, :, 1]
        yield img[..., np.newaxis], mask[..., np.newaxis]


def npy_dataset(maindir, shape_i, shape_m, seed: int = 2, batch: int = 1):
    ds = tf.data.Dataset.from_generator(lambda: npy_loader(maindir=maindir, seed = seed), 
                                        output_types=(tf.float16, tf.float16), 
                                        output_shapes=(shape_i, shape_m))
    return ds.batch(batch)


batch_size = 16

train_df = npy_dataset('/home/careinfolab/unet_mammo/images/pos_norm_elli/train/', 
                       (1024, 832, 1),
                       (1024, 832, 1),
                       seed=2, 
                       batch=batch_size)

val_df = npy_dataset('/home/careinfolab/unet_mammo/images/pos_norm_elli/val/', 
                       (1024, 832, 1),
                       (1024, 832, 1),
                       seed=2, 
                       batch=batch_size)

test_df = npy_dataset('/home/careinfolab/unet_mammo/images/pos_norm_elli/test/', 
                       (1024, 832, 1),
                       (1024, 832, 1),
                       seed=2, 
                       batch=batch_size)

bn = bisenet.BiSeNetV2((1024, 832, 1), 1)
model = bn.build_graph()
model.compile(optimizer=Adam(), loss=[seg_loss], metrics=[dice_coef, iou_coef])
name = './saved_models/bisenetv2_test1_elli'

model_checkpoint = ModelCheckpoint(name, monitor='val_loss', save_best_only=True)

history = model.fit(train_df, epochs=150, verbose=1, shuffle=True,
                    validation_data=val_df,
                    callbacks=[model_checkpoint])