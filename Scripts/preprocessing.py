# preprocessing.py

import tensorflow as tf
import os
from tensorflow.keras import layers, models

def load_datasets(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    # Data augmentation for training
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    ).map(lambda x, y: (tf.image.random_flip_left_right(x), y))

    # Validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
