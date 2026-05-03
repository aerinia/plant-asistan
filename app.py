import streamlit as st
import json
from PIL import Image
from model import load_model, run_inference
from utils import generate_caption, build_prompt, extract_json_from_response

st.set_page_config(page_title="Smart Agri Assistant", page_icon="🌱", layout="centered")

st.markdown("""
<style>
.title {
    color: #4ade80;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🌱 Smart Agri Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered plant disease diagnosis using Gemma 4</p>", unsafe_allow_html=True)

# Load model (cached to avoid reloading on every interaction)
@st.cache_resource
def get_model_and_tokenizer():
    return load_model()

# Language selector
lang = st.selectbox("Language / Dil", ["EN", "TR"])

# Language text dictionary
text_ui = {
    "EN": {
        "upload": "Upload a picture of the diseased plant leaf",
        "analyze": "🔍 Analyze Plant",
        "loading": "Analyzing image and generating diagnosis...",
        "json_title": "### Structured Diagnosis (JSON)",
        "result_title": "### Formatted Diagnosis",
        "disease": "Disease",
        "confidence": "Confidence",
        "symptoms": "Symptoms",
        "treatment": "Treatment",
        "severity": "Severity"
    },
    "TR": {
        "upload": "Hasta bitki yaprağının bir fotoğrafını yükleyin",
        "analyze": "🔍 Bitkiyi Analiz Et",
        "loading": "Görüntü inceleniyor ve teşhis oluşturuluyor...",
        "json_title": "### Yapılandırılmış Teşhis (JSON)",
        "result_title": "### Formatlanmış Teşhis",
        "disease": "Hastalık",
        "confidence": "Güvenilirlik",
        "symptoms": "Belirtiler",
        "treatment": "Tedavi",
        "severity": "Ciddiyet"
    }
}
ui = text_ui[lang]

# Image uploader
uploaded_file = st.file_uploader(ui["upload"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button(ui["analyze"], type="primary"):
        with st.spinner(ui["loading"]):
            try:
                # 1. Load Model
                model, tokenizer = get_model_and_tokenizer()
                
                # 2. Image to Caption (Placeholder logic as requested)
                caption = generate_caption(image)
                
                # 3. Build Prompt (Zero-shot instruction for JSON)
                prompt = build_prompt(caption, language=lang)
                
                # 4. Run Inference
                raw_response = run_inference(model, tokenizer, prompt)
                
                # 5. Extract JSON Output
                result_json = extract_json_from_response(raw_response)
                
                # Output Results
                st.markdown(ui["json_title"])
                st.json(result_json)
                
                if "error" not in result_json:
                    st.markdown(ui["result_title"])
                    st.success(f"**{ui['disease']}**: {result_json.get('disease', 'N/A')}")
                    st.info(f"**{ui['confidence']}**: {result_json.get('confidence', 'N/A')}")
                    st.warning(f"**{ui['symptoms']}**: {result_json.get('symptoms', 'N/A')}")
                    st.error(f"**{ui['treatment']}**: {result_json.get('treatment', 'N/A')}")
                    st.info(f"**{ui['severity']}**: {result_json.get('severity', 'N/A').upper()}")
                else:
                    st.error("Failed to parse the model output into valid JSON.")
                    st.text(result_json.get("raw_output", ""))
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
