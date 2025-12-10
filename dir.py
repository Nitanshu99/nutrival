import os
from pathlib import Path

def create_directory_structure():
    """
    Creates the NutriVal project directory structure from inside the nutrival folder.
    Skips creation if directories or files already exist.
    """
    
    # Base directory is current directory
    base_dir = Path(".")
    
    # Define the complete directory structure
    directories = [
        "data/images",
        "data/nutrition",
        "data/chroma_db",
        "data/models",
        "src/offline",
        "src/training",
        "src/vision",
        "src/logic",
        "src/rag",
        "src/genai",
    ]
    
    # Define files to create (empty files as placeholders)
    files = [
        "data/nutrition/nutrients.parquet",
        "src/offline/optimize_text.py",
        "src/offline/build_index.py",
        "src/training/train_classifier.py",
        "src/vision/efficientnet.py",
        "src/vision/clip_vision.py",
        "src/logic/router.py",
        "src/logic/decomposer.py",
        "src/rag/retriever.py",
        "src/rag/reranker.py",
        "src/rag/graph.py",
        "src/genai/insights.py",
        "app.py",
        "requirements.txt",
        "README.md",
    ]
    
    print(f"Creating directory structure in: {base_dir.absolute()}\n")
    
    # Create directories
    print("Creating directories:")
    for directory in directories:
        dir_path = base_dir / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}/")
        else:
            print(f"○ Exists: {directory}/")
    
    # Create files (empty placeholder files)
    print("\nCreating files:")
    for file in files:
        file_path = base_dir / file
        if not file_path.exists():
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Create empty file
            file_path.touch()
            print(f"✓ Created: {file}")
        else:
            print(f"○ Exists: {file}")
    
    # Create __init__.py files for Python packages
    print("\nCreating __init__.py files:")
    init_dirs = [
        "src",
        "src/offline",
        "src/training",
        "src/vision",
        "src/logic",
        "src/rag",
        "src/genai",
    ]
    
    for init_dir in init_dirs:
        init_file = base_dir / init_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✓ Created: {init_dir}/__init__.py")
        else:
            print(f"○ Exists: {init_dir}/__init__.py")
    
    print("\n✅ Directory structure created successfully!")
    print(f"\nProject root: {base_dir.absolute()}")

if __name__ == "__main__":
    create_directory_structure()
    