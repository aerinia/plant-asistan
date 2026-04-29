from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import time

# NOTE: Running a 4B model on CPU is extremely slow. 
# For the demo, we recommend running this on Google Colab (Option B).

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device.upper()}")

print("Model yükleniyor... (Bu işlem CPU'da 5-10 dakika sürebilir)")
start_time = time.time()

try:
    # Use float16 if GPU is available, otherwise stick to float32 for CPU stability
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    base = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-4b-it",
        torch_dtype=dtype,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")
    
    print("LoRA ağırlıkları yükleniyor...")
    model = PeftModel.from_pretrained(base, "./tarim-lora")
    model.eval()
    
    load_time = time.time() - start_time
    print(f"Model hazır! (Yükleme süresi: {load_time:.1f} saniye)")
    
    prompt = """<start_of_turn>user
You are an expert agricultural assistant.
Plant: Tomato
Condition observed: Early blight
What is your diagnosis and recommendation?<end_of_turn>
<start_of_turn>model
"""
    
    print("\nAnaliz başlatılıyor...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    
    print("\nCevap:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1])

except Exception as e:
    print(f"\nHata oluştu: {str(e)}")
    print("İpucu: RAM yetersiz olabilir. Model 16GB+ RAM gerektirir.")