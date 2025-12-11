"""
Module for defining the data augmentation pipeline.
"""
import tensorflow as tf
import keras
from keras import layers

def get_augmentation_layer() -> keras.Sequential:
    """
    Creates a Keras Sequential layer for data augmentation.
    
    Includes:
    - Random Horizontal Flip
    - Random Rotation (+/- 45 degrees)
    - Custom Gaussian Blur (5x5 kernel, Sigma ~1.0)

    Returns:
        keras.Sequential: A configured Keras preprocessing model.
    """
    
    def apply_gaussian_blur(images: tf.Tensor) -> tf.Tensor:
        """
        Applies a 5x5 Gaussian approximation blur to a batch of images.
        Nested helper to maintain one-function-per-file constraint.
        """
        # 5x5 Gaussian kernel approximation (Sigma ~1.0)
        kernel = tf.constant([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=tf.float32)
        
        # Use tf.math.divide to fix operator "/" issues with Pylance
        kernel = tf.math.divide(kernel, 256.0)
        
        # Reshape for depthwise_conv2d: [height, width, in_channels, channel_multiplier]
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        
        return tf.nn.depthwise_conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')

    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        # 0.125 * 2pi (or 360) = 45 degrees
        layers.RandomRotation(0.125), 
        layers.Lambda(apply_gaussian_blur)
    ])