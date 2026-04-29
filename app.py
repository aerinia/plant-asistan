import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import streamlit as st
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image

# Sayfa Yapılandırması
st.set_page_config(page_title="🌱 Smart Agri-Assistant", page_icon="📸", layout="wide")

# CSS Stilleri
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stButton>button { 
        width: 100%; 
        border-radius: 10px; 
        height: 3em; 
        background-color: #2e7d32; 
        color: white; 
        font-weight: bold; 
        font-size: 16px;
    }
    .diagnosis-card { 
        background-color: #f1f8e9; 
        color: #1b5e20; 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 8px solid #2e7d32; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        margin-top: 20px;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# ÇOK DİLLİ SÖZLÜK
LANGUAGES = {
    "TR": {
        "title": "📸 Akıllı Tarım Asistanı",
        "desc": "Bitkinizin hastalıklı veya sorunlu bölgesinin fotoğrafını yükleyin, asistanımız analiz etsin.",
        "upload": "Bitki Fotoğrafı Yükle",
        "note": "Eklemek istediğiniz bir not var mı? (Örn: Domates, 2 haftadır böyle)",
        "analyze": "🔍 Fotoğrafı Analiz Et",
        "loading": "Yapay Zeka fotoğrafı inceliyor, lütfen bekleyin...",
        "prompt": "Sen uzman bir ziraat mühendisisin. Bu bitki fotoğrafını dikkatlice incele. Gördüğün hastalığa mantıklı bir teşhis koy ve Türkçe olarak profesyonel bir tedavi önerisi sun.",
        "farmer_note": "Çiftçinin Notu",
        "result": "🧑‍🌾 Teşhis ve Öneri:"
    },
    "EN": {
        "title": "📸 Smart Agri-Assistant",
        "desc": "Upload a photo of your plant's diseased or problematic area for our AI to analyze.",
        "upload": "Upload Plant Photo",
        "note": "Any additional notes? (e.g., Tomato, it's been like this for 2 weeks)",
        "analyze": "🔍 Analyze Photo",
        "loading": "AI is analyzing the photo, please wait...",
        "prompt": "You are an expert agricultural engineer. Carefully examine this plant photo. Provide a logical diagnosis for the disease you see and offer a professional treatment recommendation in English.",
        "farmer_note": "Farmer's Note",
        "result": "🧑‍🌾 Diagnosis & Recommendation:"
    },
    "AR": {
        "title": "📸 مساعد الزراعة الذكي",
        "desc": "قم بتحميل صورة للمنطقة المريضة أو التي بها مشكلة في نباتك ليقوم الذكاء الاصطناعي بتحليلها.",
        "upload": "تحميل صورة النبات",
        "note": "أي ملاحظات إضافية؟ (مثال: طماطم، كانت هكذا منذ أسبوعين)",
        "analyze": "🔍 تحليل الصورة",
        "loading": "الذكاء الاصطناعي يقوم بتحليل الصورة، يرجى الانتظار...",
        "prompt": "أنت مهندس زراعي خبير. افحص صورة هذا النبات بعناية. قدم تشخيصًا منطقيًا للمرض الذي تراه وقدم توصية علاجية احترافية باللغة العربية.",
        "farmer_note": "ملاحظة المزارع",
        "result": "🧑‍🌾 التشخيص والتوصية:"
    }
}

@st.cache_resource
def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_name = "unsloth/gemma-3-4b-it" 
    
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    model_vocab_size = base.get_input_embeddings().weight.shape[0]
    if len(processor.tokenizer) > model_vocab_size:
        base.resize_token_embeddings(len(processor.tokenizer))
        
    base.eval()
    return base, processor

model, processor = load_model()

# Arayüz Elemanları
lang_choice = st.selectbox("🌐 Language / Dil / لغة", ["TR", "EN", "AR"])
L = LANGUAGES[lang_choice]

st.title(L["title"])
st.write(L["desc"])

uploaded_file = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"])
user_note = st.text_input(L["note"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Photo", width=400)

    if st.button(L["analyze"]):
        with st.spinner(L["loading"]):
            prompt_text = L["prompt"]
            if user_note:
                prompt_text += f"\n{L['farmer_note']}: {user_note}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=300,
                    temperature=0.3, 
                    top_p=0.9,
                    do_sample=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            result = processor.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            st.markdown(f"<div class='diagnosis-card'><strong>{L['result']}</strong><br><br>{result.strip()}</div>", unsafe_allow_html=True)