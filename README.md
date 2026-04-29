# 🌱 Smart Agri-Assistant

Smart Agri-Assistant is an LLM-based, multimodal web application designed to help farmers and agricultural engineers diagnose plant diseases through visual analysis. By uploading an image of an affected plant, users receive instant, professional diagnoses and treatment recommendations.

The project leverages state-of-the-art vision-language models and is optimized for low-memory environments. It supports three languages: English, Turkish, and Arabic.

## 🚀 Features

* **Multimodal Visual Analysis:** Processes uploaded images of crops alongside optional text notes.
* **Zero-Shot Diagnosis:** Utilizes the visual understanding capabilities of the Gemma 3 model without relying on external APIs.
* **Dynamic i18n Support:** The system automatically adjusts its internal prompt and UI language based on the user's selection (EN/TR/AR).
* **Optimized Inference:** Employs 4-bit dynamic quantization to run heavy LLMs seamlessly within constrained environments (like Google Colab).
* **Interactive UI:** Built with Streamlit for a clean, accessible, and user-friendly experience.

## 🛠️ Technology Stack

* **Model:** `unsloth/gemma-3-4b-it` (Gemma 3 4B Instruct)
* **Framework:** Streamlit
* **Deep Learning:** PyTorch, Transformers, Accelerate
* **Optimization:** BitsAndBytes (4-bit Quantization, bfloat16)
* **Image Processing:** AutoProcessor, Pillow

## ⚙️ Installation

To run this application locally or in a cloud environment (e.g., Google Colab), ensure Python 3.8+ is installed, then follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/aerinia/plant-asistan](https://github.com/aerinia/plant-asistan
   cd plant-asistan
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```