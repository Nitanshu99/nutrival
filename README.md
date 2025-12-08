# 🥗 Nutrival: Automated Nutritional Analysis App

**Course:** Data Analytics-3 (Final Project)  
**Type:** Individual Project Implementation  
**Cuisines:** Indian & Chinese (East vs. South)  
**Instructor:** Prof. Dr. Gayan de Silva

---

## 📖 1. Overview
**Nutrival** is an AI-powered dietary assessment application designed to run locally with minimal resources. It automatically identifies food from images, calculates precise nutritional values based on user-defined portion sizes, and generates culinary insights using a local Large Language Model (LLM).

This project integrates **Computer Vision (EfficientNet)** for classification, **Fuzzy Logic (Levenshtein Distance)** for multi-dataset retrieval, and **Generative AI (DistilGPT2)** for user engagement.

---

## 📂 2. Directory Structure & Modularity
To comply with the **"One File = One Function"** modular design requirement, the codebase is strictly separated by responsibility. The data is organized into specific subdirectories for images, nutrition databases, and models.

```text
nutrival/
│
├── data/
│   ├── images/                # Training Image Datasets
│   │   ├── indian_food/       # 80 classes (Indian Food Images Dataset)
│   │   └── chinese_food/      # 175 classes (Chinese Food 175 Dataset)
│   │
│   ├── nutrition/             # Nutrition Databases (CSVs)
│   │   ├── indian_nutrition.csv   # Source: Indian Food Nutritional Values (2025)
│   │   └── chinese_nutrition.csv  # Source: Common Chinese Food Dataset
│   │
│   └── models/                # Saved Model Artifacts
│       └── efficientnet_b0.h5     # The trained Vision Model
│
├── src/                       # Source Code - Modular Architecture
│   ├── ingestion/
│   │   └── load_dataset.py    # Func: load_dynamic_data() -> Scans data/images & builds TF dataset
│   │
│   ├── training/
│   │   └── train_model.py     # Func: execute_training() -> Fine-tunes EfficientNet, saves to data/models
│   │
│   ├── inference/
│   │   └── predict_class.py   # Func: predict_image() -> Returns class string from pixels
│   │
│   ├── nutrition/
│   │   ├── load_db.py         # Func: load_all_csvs() -> Reads both Indian & Chinese CSVs into memory
│   │   ├── search_db.py       # Func: find_best_match() -> Runs Levenshtein on merged DB keys
│   │   └── calculator.py      # Func: calculate_macros() -> Applies weight math to 100g base
│   │
│   └── genai/
│       └── generator.py       # Func: generate_insight() -> Runs local DistilGPT2 inference
│
├── app.py                     # Main Streamlit Entry Point (UI Layout only)
├── classes.json               # Auto-generated list of the 255 supported dishes
├── requirements.txt           # Python dependencies
└── README.md                  # Project Documentation
```

---

## 🛠️ 3. Architecture & Logic Flow

The application follows a linear 5-step pipeline:

1. **Input:** User uploads an image and selects a portion weight (e.g., 250g) via the UI.
2. **Vision Engine (`predict_class.py`):**
   * **Model:** EfficientNetB0 (Transfer Learning) stored in `data/models/`.
   * **Task:** Classifies the image into one of 255 categories (80 Indian + 175 Chinese).
3. **Retrieval Engine (`search_db.py`):**
   * **Data Sources:** Loads both `indian_nutrition.csv` and `chinese_nutrition.csv` from `data/nutrition/`.
   * **Logic:** Uses **Levenshtein Distance** to search across both files simultaneously. It finds the closest matching dish name (e.g., linking the vision label "butter_naan" to the CSV entry "Naan, Butter").
4. **Math Engine (`calculator.py`):**
   * **Logic:** Retrieves "per 100g" values from the matched row and scales them: `(User_Weight / 100) * Base_Value`.
5. **GenAI Engine (`generator.py`):**
   * **Model:** `distilgpt2` (HuggingFace Transformers).
   * **Task:** Generates a 1-sentence cultural fact, description, or simple healthy alternative for the identified dish.

---

## 🚀 4. Installation & Setup

### Prerequisites

* Python 3.12.9
* ~500MB Disk Space (for models and data)

### Step 1: Install Dependencies

```bash
pip install tensorflow streamlit pandas transformers torch python-Levenshtein
```

### Step 2: Prepare Data

1. **Images:**
   * Download **Indian Food Images Dataset** and **Chinese Food 175**.
   * Extract them into `data/images/` so that the folder contains all ~255 dish subfolders.
2. **Nutrition:**
   * Download **Indian Food Nutritional Values Dataset (2025)** and place it in `data/nutrition/` as `indian_nutrition.csv`.
   * Download **Common Chinese Food Dataset** (Nutrition component) and place it in `data/nutrition/` as `chinese_nutrition.csv`.

### Step 3: Train the Vision Model

Run the training module once. This will save the trained model to the `data/models/` directory:

```bash
python src/training/train_model.py
```

### Step 4: Run the Application

Launch the web interface:

```bash
streamlit run app.py
```

---

## 📱 5. User Guide

1. **Launch App:** Open the URL provided by Streamlit (usually `localhost:8501`).
2. **Upload:** Click "Browse files" and select a clear image of an Indian or Chinese dish.
3. **Adjust Portion:** Use the slider or input box to set your estimated portion size in grams (default is 100g).
4. **Analyze:** The app will display:
   * **Predicted Name:** What the AI thinks the food is.
   * **Nutrition Table:** Calories, Protein, Fat, and Carbs calculated for your specific portion.
   * **AI Insight:** A generated description or tip about the meal.

---

## 🌟 6. Features

### Core Requirements (Met)

* **Pretrained Vision Model:** Uses EfficientNetB0.
* **Local Dataset:** Fine-tuned on merged Indian & Chinese datasets (Local Cuisine customization).
* **Nutrition Retrieval:** Queries specific CSV datasets using fuzzy matching logic.
* **GenAI Integration:** Uses local `distilgpt2` for explanations and insights.
* **UI:** Simple Streamlit interface for upload and analysis.

### Extended Features (Bonus)

* **Portion-Size Estimation:** Users interactively adjust weight (grams) to see real-time macro updates.
* **Multi-Dataset Search:** The Levenshtein algorithm dynamically bridges the gap between different dataset naming conventions (e.g., `data/nutrition/indian_nutrition.csv` vs `data/nutrition/chinese_nutrition.csv`).
* **Modular Design:** Strict separation of concerns (1 file = 1 function), ensuring code quality and reproducibility.

---

## ⚠️ 7. Limitations

* **Dataset Bias:** The model is strictly limited to the 255 classes it was trained on. Foods outside this list (e.g., Western or African cuisine) will be misclassified as the closest Indian/Chinese equivalent.
* **GenAI Hallucinations:** As `distilgpt2` is a small model, it may occasionally generate generic or repetitive text compared to larger models like GPT-4.
* **Single Item Detection:** The current vision model assumes one dominant food item per image and does not support multi-food object detection (segmentation).

---

## 🔮 8. Future Roadmap

* **User Correction Loop:** Allow users to manually correct the predicted label if the AI is wrong, improving database matching.
* **Multi-Food Detection:** Implement YOLO or Mask R-CNN to identify multiple dishes on a single 'Thali' or plate.
* **Dietary Personalization:** Integrate user profiles (allergies, goals) to provide personalized warnings (e.g., "High Sodium warning for your profile").