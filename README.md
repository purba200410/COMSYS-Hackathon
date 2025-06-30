# COMSYS Hackathon – Gender Classification & Face Verification

This repository contains the source code for:

- **Task A:** Gender Classification
- **Task B:** Face Verification (Matching distorted faces with identity folders)

## 🧠 Task A – Gender Classification

- **Goal:** Classify faces as `male` or `female`
- **Model:** CNN-based image classifier
- **Training Script:** `train.py`
- **Input:** Preprocessed face images (224×224)

## 🧑‍🤝‍🧑 Task B – Face Verification

- **Goal:** Verify if a distorted image belongs to a given identity folder
- **Model:** CNN-based embedding model with cosine similarity
- **Training Script:** `train_face.py`
- **Testing Script:** `test_face.py`

## 📁 Project Structure
COMSYS-Hackathon/
├── train.py # Task A training script
├── train_face.py # Task B training script
├── test_face.py # Task B evaluation script
├── model.py # Shared model architecture
├── augmentations.py # Data augmentations
├── requirements.txt # Dependencies
├── README.md # Project overview

markdown
Copy
Edit

⚠️ **Note:** Dataset folders and trained `.pth` models are not included due to size. Please use your own dataset or request access from the organizers.

## ✅ Setup Instructions

1. Clone the repo:
    ```bash
    git clone https://github.com/purba200410/COMSYS-Hackathon.git
    cd COMSYS-Hackathon
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your datasets in the appropriate folders (`Task_B/`, `dataset/`) if needed.

4. Run the training or evaluation scripts:
    ```bash
    python train.py          # Task A
    python train_face.py     # Task B - Training
    python test_face.py      # Task B - Evaluation
    ```

## 🛠️ Tools Used

- Python 3.13.5
- PyTorch
- torchvision
- scikit-learn
- tqdm
- PIL (Pillow)

## 📫 Author

- GitHub: [@purba200410](https://github.com/purba200410)

