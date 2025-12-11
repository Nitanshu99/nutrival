"""
Module for loading, combining, and formatting the nested food datasets.
"""
import os
from typing import cast
import keras
import tensorflow as tf

def load_and_combine_datasets(data_dir: str, img_size: tuple[int, int], batch_size: int):
    """
    Loads Chinese and Indian food datasets, merges them, and formats labels.
    
    RETURNS: Full dataset (no validation split) and class names.
    """
    chinese_dir = os.path.join(data_dir, 'chinese_food')
    indian_dir = os.path.join(data_dir, 'indian_food')

    print(f"Loading ALL Chinese data from {chinese_dir}...")
    # Load all Chinese images (no validation split)
    # Pylance incorrectly infers this as 'list', so we explicitly cast to tf.data.Dataset
    c_ds = cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        chinese_dir,
        label_mode='int',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    ))
    # class_names is a dynamic attribute added by Keras, so we ignore the type check
    c_classes = c_ds.class_names # type: ignore
    
    print(f"Loading ALL Indian data from {indian_dir}...")
    # Load all Indian images (no validation split)
    i_ds = cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        indian_dir,
        label_mode='int',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    ))
    i_classes = i_ds.class_names # type: ignore

    # --- Label Shifting & Merging ---
    offset = len(c_classes)
    
    def shift_label(image, label):
        return image, label + offset

    # Apply shift to Indian dataset
    i_ds_shifted = i_ds.map(shift_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Concatenate datasets
    full_ds = c_ds.concatenate(i_ds_shifted)

    # Shuffle the combined data heavily to mix cuisines
    full_ds = full_ds.shuffle(buffer_size=1000, seed=123).prefetch(tf.data.AUTOTUNE)

    # --- Name Formatting ---
    raw_classes = c_classes + i_classes
    formatted_classes = [name.replace('_', ' ').title() for name in raw_classes]
    
    print(f"Total classes merged: {len(formatted_classes)}")
    
    return full_ds, formatted_classes