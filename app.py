import streamlit as st
import time
from src.model import FakeNewsModel
import os

# Sayfa ayarları - Bunu en başa koymalıyız
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Premium Tasarım için CSS Eklentisi (Glassmorphism & animasyonlar)
st.markdown("""
<style>
    /* Ana arka planı yumuşak bir degrade ile ayarlama */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* Input kutuları ve form alanları için stil */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 16px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00d2ff !important;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.5) !important;
    }
    
    /* Buton tasarımı */
    .stButton>button {
        background: linear-gradient(to right, #00d2ff, #3a7bd5) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4) !important;
    }
    
    /* Uyarı ve Başarı kutuları için şeffaf cam stili */
    .stAlert {
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Dosya yükleme alanı stili */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 10px;
    }

    h1, h2, h3, p {
        font-family: 'Helvetica Neue', sans-serif !important;
    }
    
    .title-highlight {
        color: #00d2ff;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Başlık ve Açıklama
st.markdown('<h1 style="text-align: center;">Yapay Zeka Destekli <br><span class="title-highlight">Sahte Haber</span> Tespit Sistemi 📰</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a0aec0;">Metni girin veya bir belge yükleyin, yapay zeka haberin doğruluk oranını analiz etsin.</p>', unsafe_allow_html=True)
st.divider()

# Modelin yüklenmesi
@st.cache_resource
def load_model_instance():
    try:
        return FakeNewsModel()
    except Exception as e:
        return None

detector = load_model_instance()

if detector and detector.model is None:
    st.warning("⚠️ Eğitilmiş bir model bulunamadı! Lütfen önce verisetini indirip `data/True.csv` ve `data/Fake.csv` olarak projenin içine koyduktan sonra terminalde `python src/model.py` çalıştırın.")
    st.stop()

# Sekme Yapısı
tab1, tab2 = st.tabs(["📝 Metin Analizi", "📁 Dosyadan Analiz"])

def display_prediction(text_input):
    """Tahmin sonucunu animasyonları ile ekrana basar"""
    if len(text_input.strip()) < 15:
        st.error("Lütfen daha uzun ve anlamlı bir metin girin.")
        return

    # Animasyonlu yükleme barı
    progress_text = "Yapay Zeka Metni Analiz Ediyor..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
        
    time.sleep(0.5)
    my_bar.empty()
    
    # Model Tahmini
    try:
        label, conf = detector.predict(text_input)
        
        st.markdown("### 📊 Analiz Sonucu")
        if label == "Real News":
            st.success(f"✅ **GERÇEK HABER** \n\n Model Güven Oranı: **% {conf:.2f}**")
            st.balloons()
        else:
            st.error(f"🚨 **SAHTE HABER (Fake News)** \n\n Model Güven Oranı: **% {conf:.2f}**")
            
        with st.expander("Metnin İşlenmiş (Temizlenmiş) Halini Gör"):
            cleaned = detector.preprocessor.clean_text(text_input)
            st.write(cleaned)

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


# TAB 1: YAZI İLE
with tab1:
    user_text = st.text_area("Analiz Edilecek Haber Metnini Yapıştırın:", height=200, placeholder="Haber metni buraya...")
    
    if st.button("🚀 Haberi Analiz Et", key="btn_text"):
        display_prediction(user_text)

# TAB 2: DOSYA İLE
with tab2:
    uploaded_file = st.file_uploader("Bir metin (.txt) dosyası yükleyin:", type=["txt"])
    
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("Yüklenen Dosya İçeriği (Önizleme):", value=file_content[:500] + "..." if len(file_content) > 500 else file_content, height=150, disabled=True)
        
        if st.button("🚀 Yüklenen Dosyayı Analiz Et", key="btn_file"):
            display_prediction(file_content)

st.divider()
st.markdown('<p style="text-align: center; color: #a0aec0; font-size: 12px;">© 2026 Fake News Detection AI. Tüm hakları saklıdır.</p>', unsafe_allow_html=True)
