import shap
import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)

#Initialize SHAP
shap.initjs()

def model_predict(texts, model, tokenizer, device="cuda"):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu().numpy()

def explain_ner_model(model_path, tokenizer_path, sample_text, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def f(texts):
        return model_predict(texts, model, tokenizer, device)
    
    explainer = shap.Explainer(f, tokenizer)
    
    shap_values = explainer([sample_text])
    
    preds = f([sample_text])
    pred_labels = [label_map[np.argmax(pred)] for pred in preds[0][1:-1]]
    
    # Visualization
    print(f"\nModel: {model_path.split('/')[-1]}")
    print(f"Sample text: {sample_text}")
    print(f"Predicted labels: {pred_labels}")
    
    #force plot for each predicted entity
    for i, (token, label) in enumerate(zip(tokenizer.tokenize(sample_text), pred_labels)):
        if label != "O":  # Only show explanations for actual entities
            print(f"\nToken: {token} | Label: {label}")
            shap.plots.text(shap_values[:,:,i])
    
    return shap_values

label_map = {
    0: "O",
    1: "B-PRODUCT",
    2: "I-PRODUCT",
    3: "B-PRICE",
    4: "I-PRICE",
    5: "B-LOCATION",
    6: "I-LOCATION"
}

sample_amharic_text = "በአዲስ አበባ ላይ አዲስ ስልክ በ 5000 ብር ይገኛል"

xlmr_shap = explain_ner_model(
    "C:/Users/hp/Desktop/Kifya/week_4/model",
    "C:/Users/hp/Desktop/Kifya/week_4/tokenizer",
    sample_amharic_text,
    label_map
)

distilbert_shap = explain_ner_model(
    "C:/Users/hp/Desktop/Kifya/week_4/distilbert_model",
    "C:/Users/hp/Desktop/Kifya/week_4/distilbert_tokenizer",
    sample_amharic_text,
    label_map
)

mbert_shap = explain_ner_model(
    "C:/Users/hp/Desktop/Kifya/week_4/mbert_model",
    "C:/Users/hp/Desktop/Kifya/week_4/mbert_tokenizer",
    sample_amharic_text,
    label_map
)