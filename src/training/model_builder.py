"""
Module for constructing the EfficientNet architecture.
"""
import keras
from keras import layers, applications, Model

def build_specialist_model(num_classes: int, augmentation_layer: keras.layers.Layer) -> Model:
    """
    Builds the Specialist Vision Model using EfficientNetB0.
    """
    print("Building EfficientNetB0 model...")
    
    # Load EfficientNetB0 from keras.applications
    base_model = applications.EfficientNetB0(
        include_top=False, 
        weights="imagenet", 
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Use keras.Input directly
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Apply the augmentation layer passed from the pipeline
    x = augmentation_layer(inputs)
    
    # Pass through base model (training=False ensures BatchNormalization stays in inference mode)
    x = base_model(x, training=False)
    
    # Rebuild top layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)