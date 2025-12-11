"""
Main training orchestrator for the Nutrival Specialist Model.
"""
import json
from keras import optimizers, losses, callbacks

# Local imports
from src.training.data_loader import load_and_combine_datasets
from src.training.augmentation import get_augmentation_layer
from src.training.model_builder import build_specialist_model

def train_efficientnet():
    """
    Orchestrates the training pipeline on the FULL dataset with Early Stopping.
    """
    config = {
        "img_size": (224, 224),
        "batch_size": 32,
        "epochs": 25, # Set this high (e.g., 25), EarlyStopping will cut it short
        "data_dir": "data/images/",
        "save_model_path": "data/models/efficientnet_model.keras",
        "save_classes_path": "data/models/class_names.json"
    }

    # 1. Load All Data
    full_ds, class_names = load_and_combine_datasets(
        config["data_dir"], 
        config["img_size"], 
        config["batch_size"]
    )

    num_classes = len(class_names)
    print(f"Training on {num_classes} classes (Full Dataset).")

    # 2. Get Augmentation & Build Model
    aug_layer = get_augmentation_layer()
    model = build_specialist_model(num_classes, aug_layer)

    # 3. Compile
    # Pylance may incorrectly flag 'optimizer' as expecting a string. 
    # We suppress this with type: ignore as passing the object is valid and necessary for setting learning_rate.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3), # type: ignore
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # --- DEFINE EARLY STOPPING ---
    # Monitor 'loss' (since we don't have val_loss split). 
    # Stops if loss doesn't improve for 3 epochs.
    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    print(f"Starting training (Max {config['epochs']} epochs)...")
    
    # 4. Train with Callback
    model.fit(
        full_ds, 
        epochs=config["epochs"], 
        callbacks=[early_stopping]
    )

    # 5. Save Artifacts
    print(f"Saving model to {config['save_model_path']}...")
    model.save(config['save_model_path'])
    
    with open(config['save_classes_path'], 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2)
        
    print("Training complete.")

if __name__ == "__main__":
    train_efficientnet()