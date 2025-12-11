# 🥗 Nutrival: Automated Nutritional Analysis App

**Course:** Data Analytics-3 (Final Project)  
**Type:** Individual Project Implementation  
**Instructor:** Prof. Dr. Gayan de Silva  
**Deadline:** 16.12.2025

---

## 📖 Overview

**Nutrival** is a hybrid AI dietary assessment system designed to solve the trade-off between **precision** (identifying known local dishes) and **flexibility** (identifying generic foods).

It employs a **"Router" architecture** that directs food images to the most appropriate AI expert:

- **The Specialist (EfficientNet):** Instantly recognizes 255 specific local recipes (e.g., "Butter Chicken") trained on a curated dataset
- **The Generalist (CLIP):** Handles any unknown food (e.g., "Red Apple") using open-vocabulary vector search
- **The Reasoner (Qwen 0.5B + LangGraph):** A RAG (Retrieval-Augmented Generation) engine that intelligently maps generic ingredients to exact database entries using semantic reasoning

---

## 📂 Directory Structure

The project follows a strict **"One File = One Function"** modular design.

```
nutrival/
│
├── data/
│   ├── images/                 # Training Image Datasets (Indian/Chinese)
│   ├── nutrition/
│   │   └── nutrients.parquet   # Master DB with optimized 'description_for_clip' column
│   ├── chroma_db/              # Persisted Vector Store (ChromaDB)
│   └── models/                 # Saved EfficientNet & Qwen artifacts
│
├── src/
│   ├── offline/                # DATA OPTIMIZATION (Run Once)
│   │   ├── clean_dataset.py    # Func: clean_images() -> Scans and deletes corrupt image files
│   │   ├── optimize_text.py    # Func: rewrite_descriptions() -> LLM converts DB rows to natural captions
│   │   └── build_index.py      # Func: create_vector_store() -> Embeds captions & saves to ChromaDB
│   │
│   ├── training/               # MODEL TRAINING
│   │   ├── augmentation.py     # Func: get_augmentation_layer() -> Dynamic flip/rotate/blur pipeline
│   │   ├── data_loader.py      # Func: load_and_combine_datasets() -> Loads & merges Indian/Chinese data
│   │   ├── model_builder.py    # Func: build_specialist_model() -> Compiles EfficientNetB0
│   │   └── train_classifier.py # Func: train_efficientnet() -> Main orchestrator for training
│   │
│   ├── vision/                 # IMAGE PROCESSING
│   │   ├── efficientnet.py     # Func: predict_known_dish() -> Returns label & confidence score
│   │   └── clip_vision.py      # Func: encode_image() -> Returns vector embedding of image
│   │
│   ├── logic/                  # DECISION ENGINE
│   │   ├── router.py           # Func: route_input() -> Switches between EfficientNet and CLIP
│   │   └── decomposer.py       # Func: break_down_recipe() -> LLM lists ingredients for known dishes
│   │
│   ├── rag/                    # SMART MAPPING (LangGraph)
│   │   ├── retriever.py        # Func: semantic_search() -> Queries ChromaDB for top candidates
│   │   ├── reranker.py         # Func: context_reasoning() -> LLM selects best match (Reasoning)
│   │   └── graph.py            # Func: run_rag_flow() -> Orchestrates the Retrieve -> Rerank loop
│   │
│   └── genai/                  # FINAL OUTPUT
│       └── insights.py         # Func: generate_culture_fact() -> Generates food trivia
│
├── app.py                      # Main Streamlit UI
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 🏗️ Architecture: The "Dual Brain" Logic

The system flows through a 5-step pipeline to analyze food images:

### Step 1: Input & Pre-processing

**User Action:** Uploads image (e.g., `plate.jpg`) and sets portion size (e.g., `250g`)

**System Action:** Resizes image for both EfficientNet (224×224) and CLIP (224×224)

### Step 2: The Router

The **EfficientNet** model (fine-tuned) first attempts to classify the image:

- **If Confidence Score > 40%:** Proceed to Path A (Known Dish)
- **If Confidence Score < 40%:** Proceed to Path B (Unknown Item)

### Step 3: Execution Paths

**Path A - The Specialist:**
- Accepts the label (e.g., "Butter Chicken")
- Uses LLM to decompose the dish into generic ingredients (e.g., "Chicken", "Cream", "Tomato")

**Path B - The Generalist:**
- Passes image to CLIP
- CLIP compares image embedding against vector index to find best semantic match (e.g., "Red Delicious Apple")

### Step 4: The Smart Mapper (RAG Engine)

Connects vague ingredients to precise database IDs:

1. **Retrieve:** Search ChromaDB for Top 5 items matching the ingredient
2. **Reason (Rerank):** LangGraph prompts Qwen with context
   - *Example: "Context: Butter Chicken. Options: [Sour Cream, Heavy Cream, Coffee Creamer]. Which is best?"*
3. **Result:** Qwen selects the most appropriate match (e.g., "Heavy Cream" - ID #4502)

### Step 5: Calculation & Insight

- **Math:** Retrieves macros for the selected ID and scales by portion size
- **GenAI:** Qwen generates a fun cultural fact about the identified food

---

## 🚀 Optimization Strategy: Technical Deep Dive

A critical component of this project is the **Offline Optimization Phase** that translates the database for Vision AI compatibility.

### The Problem: "Database Speak" vs. "CLIP Speak"

CLIP was trained on natural internet captions like *"A delicious slice of pepperoni pizza"*. However, nutritional databases use scientific formats:

**Raw Database Entry:**
```
"Chicken, broiler or fryer, breast, meat only, raw"
```

This mismatch causes accuracy drops because the text lacks natural language structure and visual context.

### The Solution: LLM-Based Rewriting

Before runtime, we perform a one-time **Text Optimization** process using **Qwen 2.5-0.5B**:

**The Workflow:**

1. **Input:** Read raw row: `"Chicken, broiler... breast... raw"`
2. **Prompt Qwen:** *"Rewrite this database entry into a short, visual caption for a photo"*
3. **Output:** `"Raw meat of broiler or fryer chicken"`
4. **Store:** Save to new column `description_for_clip`

### Why This Optimizes Performance

- **Higher Cosine Similarity:** Vector embedding of visual captions is mathematically closer to image embeddings than raw database strings
- **Better Retrieval:** Ensures CLIP retrieves correct items even with obscure database naming conventions
- **Semantic Alignment:** Bridges the gap between scientific terminology and visual understanding

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.12.9
- ~4GB VRAM (for running Qwen 0.5B and CLIP simultaneously)

### Step 1: Install Dependencies

```bash
pip install tensorflow torch transformers streamlit pandas pyarrow chromadb langgraph sentence-transformers
```

### Step 2: Run Offline Optimization (Crucial)

Generate the vector index before running the app:

```bash
# 1. Convert DB strings to Natural Language
python src/offline/optimize_text.py

# 2. Embed text and save to ChromaDB
python src/offline/build_index.py
```

### Step 3: Train Vision Model

Fine-tune EfficientNet on the 255-class dataset:

```bash
python src/training/train_classifier.py
```

### Step 4: Run Application

```bash
streamlit run app.py
```

---

## ⚙️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vision Specialist | EfficientNet | Classify 255 known dishes |
| Vision Generalist | CLIP | Open-vocabulary food recognition |
| LLM Reasoning | Qwen 2.5-0.5B | Ingredient mapping & insights |
| Vector Database | ChromaDB | Semantic search |
| Orchestration | LangGraph | RAG pipeline management |
| UI Framework | Streamlit | Interactive web interface |

---

## ⚠️ Limitations

- **Processing Overhead:** The Smart Mapping reasoning step adds ~2 seconds of latency per ingredient compared to simple keyword search
- **Hardware Requirements:** Running local LLM (Qwen), Vector DB (ChromaDB), and Vision Model (EfficientNet) requires decent RAM/GPU capabilities
- **Dataset Scope:** Limited to 255 pre-trained local dishes; expanding requires retraining

---

## 📊 Performance Considerations

The hybrid architecture provides:

- **High Precision:** 90%+ accuracy on known dishes (via EfficientNet)
- **High Flexibility:** Open-vocabulary recognition for any food item (via CLIP)
- **Contextual Intelligence:** Smart ingredient mapping reduces false positives (via RAG)

---

## 🎯 Future Enhancements

- Multi-food detection in single images
- Real-time meal tracking integration
- Expanded dish database beyond 255 classes
- Mobile app deployment
- User preference learning system

---

## 📝 License

This project is submitted as part of the Data Analytics-3 course requirements.

---

## 👨‍💻 Author

Individual Project Implementation  
**Course:** Data Analytics-3  
**Institution:** [Your Institution Name]  
**Submission Date:** December 16, 2025