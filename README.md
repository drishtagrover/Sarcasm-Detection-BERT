🤖Sarcasm Detection using BERT

- This project focuses on detecting sarcasm in textual data using a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. Sarcasm detection is a challenging task in Natural Language Processing (NLP) as it requires a deep understanding of context, tone, and subtle language cues. Leveraging BERT's contextual embedding capabilities significantly enhances the accuracy of sarcasm detection models.

📌 Table of Contents
1. About the Project
2. Tech Stack
3. Dataset
4. Model Architecture
5. How to Run
6. Results
7. Future Work
8. Contributors

🔍 About the Project
- The aim of this project is to build a deep learning model that classifies text as sarcastic or not sarcastic. The project uses the BERT-base-uncased model from Hugging Face Transformers and is fine-tuned on a labeled dataset of sarcastic and non-sarcastic text samples.

🛠 Tech Stack
- Python 🐍
- PyTorch ⚙️
- Hugging Face Transformers 🤗
- scikit-learn 📊
- Pandas & NumPy 📚
- Matplotlib & Seaborn 📈
- Google Colab Notebook

📂 Dataset
- The dataset used in this project contains news headlines or tweets labeled as sarcastic (1) or not sarcastic (0).
- Sample source: News Headlines Dataset For Sarcasm Detection on HuggingFace
https://huggingface.co/datasets/FlorianKibler/sarcasm_dataset_en

Preprocessing steps include:
1.Removing punctuation and stopwords
2.Tokenization using BERT tokenizer
3.Padding and truncation

🧠 Model Architecture
- Model: bert-base-uncased
- Layers: BERT + Linear Classifier
- Loss Function: Binary Cross Entropy
- Optimizer: AdamW
- Training Strategy:
Fine-tuning for several epochs
Batch size: 16/32

▶️ How to Run
- download the file and copy paste it on google colab notebook
- change your runtime environment to T4 GPU 
- kindly load the dataset from the given link

📊RESULT
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 90%   |
| Precision | 88%   |
| Recall    | 91%   |
| F1-Score  | 90%   |

<img width="617" height="218" alt="Screenshot 2025-04-10 132435" src="https://github.com/user-attachments/assets/5f74669e-a891-4459-9ee0-1fc265400365" />


<img width="961" height="186" alt="Screenshot 2025-04-10 132509" src="https://github.com/user-attachments/assets/4bf812fd-0f97-4240-9774-af6765a74c59" />


🚀 Future Work
- Deploy the model using Flask or Streamlit
- Train on larger datasets (e.g., Reddit, Twitter)
- Implement multi-class sentiment detection
- Explore multilingual sarcasm detection
