import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK paketlerini (sadece ilk çalıştırmada gerekli) indiriyoruz.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextPreprocessor:
    """
    Haber metinlerini sınıflandırma algoritması için hazırlayan ön işleme (preprocessing) sınıfı.
    OOP mantığı ile modülerleştirilmiştir.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Gelen metni temizler, stop word'leri çıkarır, noktalama işaretlerini siler ve stemming yapar.
        :param text: Temizlenecek ham metin
        :return: Temizlenmiş ve algoritma için hazır hale gelmiş metin
        """
        # 1. Küçük harfe dönüştür.
        text = text.lower()
        
        # Ajans imzalarını ve lokasyon etiketlerini temizle (Örn: "washington (reuters) - ", "london (reuters) - ")
        text = re.sub(r'^[a-z0-9\s,./|\-_]+\(\s*(reuters|ap|cnn|bbc|associated\s*press)\s*\)\s*-\s*', '', text)
        # Metin içinde geçen genel ajans isimlerini de çıkaralım (ki model kelimeyi ezberlemesin)
        text = re.sub(r'\b(reuters|ap|cnn|bbc|nytimes|associated\s*press)\b', '', text)
        
        # 2. Köşeli parantez içindekileri (örn. kaynak linkleri), URL'leri ve gereksiz etiketleri Regex ile kaldır.
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        
        # 3. Noktalama işaretlerini kaldır.
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        
        # 4. Sayıları ve yeni satır karakterlerini temizle.
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\d+', '', text) # Sadece rakamları siliyoruz (örn: covid19 -> covid olur, tüm kelime silinmez)
        
        # 5. Tokenleştir, stop word'leri çıkar ve lemmatization (sözlük kökü) uygula.
        words = text.split()
        cleaned_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return " ".join(cleaned_words)
