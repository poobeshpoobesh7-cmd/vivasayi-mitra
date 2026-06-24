import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision import transforms
from PIL import Image
import numpy as np

# ---- CONFIG ----
MODEL_PATH = "cropvision_final.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    'Pepper__bell__Bacterial_spot',
    'Pepper__bell__healthy',
    'Potato__Early_blight',
    'Potato__Late_blight',
    'Potato__healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

LANGUAGES = {
    "Tamil 🌾": {
        "title": "விவசாயி மித்ரா",
        "subtitle": "உங்கள் பயிர் நோயை கண்டறியுங்கள்",
        "upload": "இலை படத்தை பதிவேற்றுங்கள் 📸",
        "uploaded": "பதிவேற்றிய படம்",
        "analyzing": "🔍 பகுப்பாய்வு செய்கிறோம்...",
        "confidence": "நம்பகத்தன்மை",
        "remedy": "### 💊 தீர்வு:",
        "top3": "### 📊 முதல் 3 முடிவுகள்:",
        "footer": "Vivasayi Mitra — Tamil Nadu விவசாயிகளுக்காக 🌱",
        "unknown": "⚠️ இந்த பயிர் அடையாளம் காண முடியவில்லை. மிளகாய், உருளைக்கிழங்கு அல்லது தக்காளி இலையை பயன்படுத்தவும்.",
        "supported": "### 🌿 ஆதரிக்கப்படும் பயிர்கள்:",
        "crops": "🌶️ மிளகாய் | 🥔 உருளைக்கிழங்கு | 🍅 தக்காளி",
        "select_lang": "மொழி தேர்வு",
    },
    "English 🌍": {
        "title": "Vivasayi Mitra",
        "subtitle": "Detect Your Crop Disease",
        "upload": "Upload Leaf Image 📸",
        "uploaded": "Uploaded Image",
        "analyzing": "🔍 Analyzing...",
        "confidence": "Confidence",
        "remedy": "### 💊 Remedy:",
        "top3": "### 📊 Top 3 Results:",
        "footer": "Vivasayi Mitra — For Tamil Nadu Farmers 🌱",
        "unknown": "⚠️ Crop not recognized. Please use Pepper, Potato or Tomato leaves.",
        "supported": "### 🌿 Supported Crops:",
        "crops": "🌶️ Pepper | 🥔 Potato | 🍅 Tomato",
        "select_lang": "Select Language",
    }
}
DISEASE_NAMES = {
    "Tamil 🌾": {
        'Pepper__bell__Bacterial_spot': 'மிளகாய் — பாக்டீரியா புள்ளி நோய்',
        'Pepper__bell__healthy': 'மிளகாய் — ஆரோக்கியமான இலை ✅',
        'Potato__Early_blight': 'உருளைக்கிழங்கு — ஆரம்ப கருகல் நோய்',
        'Potato__Late_blight': 'உருளைக்கிழங்கு — தாமத கருகல் நோய்',
        'Potato__healthy': 'உருளைக்கிழங்கு — ஆரோக்கியமான இலை ✅',
        'Tomato_Bacterial_spot': 'தக்காளி — பாக்டீரியா புள்ளி நோய்',
        'Tomato_Early_blight': 'தக்காளி — ஆரம்ப கருகல் நோய்',
        'Tomato_Late_blight': 'தக்காளி — தாமத கருகல் நோய்',
        'Tomato_Leaf_Mold': 'தக்காளி — இலை பூஞ்சை நோய்',
        'Tomato_Septoria_leaf_spot': 'தக்காளி — செப்டோரியா புள்ளி நோய்',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'தக்காளி — சிலந்தி பூச்சி தாக்குதல்',
        'Tomato__Target_Spot': 'தக்காளி — இலக்கு புள்ளி நோய்',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'தக்காளி — மஞ்சள் சுருள் வைரஸ்',
        'Tomato__Tomato_mosaic_virus': 'தக்காளி — மொசைக் வைரஸ்',
        'Tomato_healthy': 'தக்காளி — ஆரோக்கியமான இலை ✅',
    },
    "English 🌍": {
        'Pepper__bell__Bacterial_spot': 'Pepper — Bacterial Spot',
        'Pepper__bell__healthy': 'Pepper — Healthy Leaf ✅',
        'Potato__Early_blight': 'Potato — Early Blight',
        'Potato__Late_blight': 'Potato — Late Blight',
        'Potato__healthy': 'Potato — Healthy Leaf ✅',
        'Tomato_Bacterial_spot': 'Tomato — Bacterial Spot',
        'Tomato_Early_blight': 'Tomato — Early Blight',
        'Tomato_Late_blight': 'Tomato — Late Blight',
        'Tomato_Leaf_Mold': 'Tomato — Leaf Mold',
        'Tomato_Septoria_leaf_spot': 'Tomato — Septoria Leaf Spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato — Spider Mites',
        'Tomato__Target_Spot': 'Tomato — Target Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato — Yellow Leaf Curl Virus',
        'Tomato__Tomato_mosaic_virus': 'Tomato — Mosaic Virus',
        'Tomato_healthy': 'Tomato — Healthy Leaf ✅',
    }
}

REMEDIES_LANG = {
    "Tamil 🌾": {
        'Pepper__bell__Bacterial_spot': 'காப்பர் அடிப்படையிலான பூஞ்சாணக்கொல்லி தெளிக்கவும். நோயுற்ற இலைகளை அகற்றவும்.',
        'Pepper__bell__healthy': 'உங்கள் பயிர் ஆரோக்கியமாக உள்ளது! தொடர்ந்து கவனித்து வாருங்கள்.',
        'Potato__Early_blight': 'மேன்கோசெப் அல்லது குளோரோதலோனில் தெளிக்கவும். நோயுற்ற இலைகளை எரிக்கவும்.',
        'Potato__Late_blight': 'உடனடியாக மெட்டலாக்சில் தெளிக்கவும். மழை நேரத்தில் கவனமாக இருக்கவும்.',
        'Potato__healthy': 'உங்கள் பயிர் ஆரோக்கியமாக உள்ளது! தொடர்ந்து கவனித்து வாருங்கள்.',
        'Tomato_Bacterial_spot': 'காப்பர் ஆக்சிகுளோரைடு தெளிக்கவும். நீர்ப்பாசனம் குறைக்கவும்.',
        'Tomato_Early_blight': 'ட்ரைகோடர்மா விரிடே உயிர் பூஞ்சாணக்கொல்லி பயன்படுத்தவும்.',
        'Tomato_Late_blight': 'சைமோக்சானில் + மேன்கோசெப் தெளிக்கவும். உடனடி நடவடிக்கை அவசியம்.',
        'Tomato_Leaf_Mold': 'காற்றோட்டம் அதிகரிக்கவும். கார்பெண்டாசிம் தெளிக்கவும்.',
        'Tomato_Septoria_leaf_spot': 'கீழ் இலைகளை அகற்றவும். குளோரோதலோனில் தெளிக்கவும்.',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'வேப்பெண்ணெய் தெளிக்கவும். இலைகளை தண்ணீரால் கழுவவும்.',
        'Tomato__Target_Spot': 'அசோக்ஸிஸ்ட்ரோபின் தெளிக்கவும். பயிர் சுழற்சி பின்பற்றவும்.',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'வெள்ளை ஈக்களை கட்டுப்படுத்தவும். நோயுற்ற செடிகளை அகற்றவும்.',
        'Tomato__Tomato_mosaic_virus': 'நோயுற்ற செடிகளை உடனே அகற்றவும். கருவிகளை கிருமிநாசினியால் சுத்தம் செய்யவும்.',
        'Tomato_healthy': 'உங்கள் பயிர் ஆரோக்கியமாக உள்ளது! தொடர்ந்து கவனித்து வாருங்கள்.',
    },
    "English 🌍": {
        'Pepper__bell__Bacterial_spot': 'Spray copper-based fungicide. Remove infected leaves immediately.',
        'Pepper__bell__healthy': 'Your crop is healthy! Keep monitoring regularly.',
        'Potato__Early_blight': 'Apply Mancozeb or Chlorothalonil. Burn infected leaves.',
        'Potato__Late_blight': 'Spray Metalaxyl immediately. Be cautious during rainy season.',
        'Potato__healthy': 'Your crop is healthy! Keep monitoring regularly.',
        'Tomato_Bacterial_spot': 'Apply Copper Oxychloride. Reduce irrigation frequency.',
        'Tomato_Early_blight': 'Use Trichoderma viride bio-fungicide.',
        'Tomato_Late_blight': 'Spray Cymoxanil + Mancozeb. Immediate action required.',
        'Tomato_Leaf_Mold': 'Improve air circulation. Apply Carbendazim.',
        'Tomato_Septoria_leaf_spot': 'Remove lower leaves. Spray Chlorothalonil.',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'Spray neem oil. Wash leaves with water.',
        'Tomato__Target_Spot': 'Apply Azoxystrobin. Follow crop rotation.',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Control whiteflies. Remove infected plants.',
        'Tomato__Tomato_mosaic_virus': 'Remove infected plants immediately. Sterilize tools.',
        'Tomato_healthy': 'Your crop is healthy! Keep monitoring regularly.',
    }
}

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model = efficientnet_v2_s()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 15)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- PREDICT ----
def predict(image, model):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    return CLASS_NAMES[pred_idx.item()], confidence.item(), probs[0].cpu().numpy()

# ---- UI ----
# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Vivasayi Mitra 🌾",
    page_icon="🌱",
    layout="centered"
)

# ---- SIDEBAR LANGUAGE SELECTOR ----
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/seedling.png", width=80)
    st.markdown("## விவசாயி மித்ரா")
    st.markdown("---")
    
    lang = st.radio(
        "🌐 Language / மொழி",
        options=["Tamil 🌾", "English 🌍"],
        index=0
    )
    
    st.markdown("---")
    L = LANGUAGES[lang]
    st.markdown(L["supported"])
    st.info(L["crops"])
    st.markdown("---")
    st.caption("v1.0 | EfficientNetV2-S | 90%+ Accuracy")

L = LANGUAGES[lang]

# ---- MAIN UI ----
model = load_model()

st.title(f"🌾 {L['title']}")
st.subheader(L["subtitle"])
st.markdown("---")

uploaded = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption=L["uploaded"], width=300)

    with st.spinner(L["analyzing"]):
        pred_class, confidence, all_probs = predict(image, model)

    st.markdown("---")

    # Unknown crop check
    if confidence < 0.70:
        st.warning(L["unknown"])
    else:
        disease_name = DISEASE_NAMES[lang][pred_class]
        remedy = REMEDIES_LANG[lang][pred_class]

        if "healthy" in pred_class:
            st.success(f"✅ {disease_name}")
        else:
            st.error(f"⚠️ {disease_name}")

        st.metric(L["confidence"], f"{confidence*100:.1f}%")

        st.markdown(L["remedy"])
        st.info(remedy)

        st.markdown(L["top3"])
        top3_idx = np.argsort(all_probs)[::-1][:3]
        for idx in top3_idx:
            class_key = CLASS_NAMES[idx]
            name = DISEASE_NAMES[lang][class_key]
            st.progress(
                float(all_probs[idx]),
                text=f"{name} — {all_probs[idx]*100:.1f}%"
            )

st.markdown("---")
st.caption(L["footer"])