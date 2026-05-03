# Smart Agri Assistant

A complete, minimal, and technically correct AI project for the Kaggle "Gemma 4 Good Hackathon".

## Description
Smart Agri Assistant is a prototype that analyzes plant diseases from an uploaded image and returns a structured diagnosis using a Gemma 4 model. It is designed to run efficiently on low-tier hardware like Google Colab T4 GPUs, leveraging 4-bit quantization and the `unsloth` framework.

## Pipeline Architecture
1. **Image Upload**: The user uploads a photo of a diseased plant leaf via a Streamlit UI.
2. **Image → Caption**: A lightweight placeholder function generates a text description of the image (simulating a Vision-Language Model).
3. **Caption → Prompt**: The description is combined with a deterministic, structured prompt using zero-shot prompting techniques, instructing the model to act as an agricultural engineer.
4. **LLM Inference**: The `unsloth/gemma-4-e4b-it` model processes the prompt and generates a strict JSON output representing the diagnosis.
5. **Display Results**: The Streamlit application parses the returned JSON and presents the disease, confidence, symptoms, treatment, and severity dynamically to the user.

## Why Gemma 4
Gemma 4 provides highly capable instruction-following characteristics in a compact size, making it ideal for running on edge devices or low-cost cloud instances. With `unsloth` and 4-bit quantization (`bitsandbytes`), we achieve significant memory savings and faster inference speed without sacrificing response structure and accuracy.

## How to Run

### Locally
Ensure you have a machine with an NVIDIA GPU and CUDA installed (recommended for decent inference speed).

1. Clone or download the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit application:
```bash
streamlit run app.py
```

### Google Colab
1. Upload the project files to a Google Colab environment.
2. Select a **T4 GPU** runtime.
3. Run the following cell to install requirements and start the app via localtunnel or ngrok:

```python
!pip install -r requirements.txt
!npm install localtunnel
!streamlit run app.py & npx localtunnel --port 8501
```
*(Once localtunnel provides a URL, you can open it to view the web app. It might ask for the endpoint IP address which you can fetch using `!curl ipv4.icanhazip.com`)*