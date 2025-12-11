"""
Script to scan the dataset and remove corrupt images that crash TensorFlow.
"""
import os
import tensorflow as tf

def clean_images(directory):
    """
    Walks through the directory, attempts to decode every image.
    Asks for confirmation before deleting corrupt files.
    """
    print(f"Scanning {directory} for corrupt files...")
    bad_files = []
    total_files = 0

    # 1. Scan Phase
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1
            
            if file.startswith('.'):
                continue
                
            try:
                # Attempt to decode to check validity
                image_bytes = tf.io.read_file(file_path)
                tf.io.decode_image(image_bytes, expand_animations=False)
            except (tf.errors.InvalidArgumentError, tf.errors.UnknownError):
                print(f"❌ Found corrupt: {file_path}")
                bad_files.append(file_path)
            except Exception: # pylint: disable=broad-except
                # Skip other read errors
                pass

    print(f"\nScan complete. Checked {total_files} files.")
    
    # 2. Confirmation Phase
    if bad_files:
        count = len(bad_files)
        print(f"⚠️ Found {count} corrupt files that will cause training to crash.")
        
        # User input for permission
        response = input(f"Do you want to delete these {count} files? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Deleting corrupt files...")
            for bad_file in bad_files:
                try:
                    os.remove(bad_file)
                    print(f"Deleted: {bad_file}")
                except OSError as e:
                    print(f"Failed to delete {bad_file}: {e}")
            print("Cleanup complete.")
        else:
            print("Operation aborted. No files were deleted.")
    else:
        print("✅ No corrupt files found. Dataset is clean.")

if __name__ == "__main__":
    clean_images("data/images")