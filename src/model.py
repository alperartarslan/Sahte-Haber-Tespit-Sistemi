import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing import TextPreprocessor

class FakeNewsModel:
    """
    Fake News tespiti için Makine Öğrenimi modelini (Logistic Regression) ve Vektörizatör'ü yöneten sınıf.
    """
    
    def __init__(self, model_path="models/model.pkl", vectorizer_path="models/vectorizer.pkl"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        
        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Önceden eğitilmiş model varsa yükle
        self.load_model()
        
    def load_model(self):
        """Diskten eğitilmiş modeli ve vektörizatörü yükler."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        else:
            self.model = None
            self.vectorizer = None

    def train_model(self, true_csv_path, fake_csv_path):
        """
        Kaggle Fake and Real News dataset dosyalarını alıp modeli eğitir ve değerlendirir.
        """
        print("Veriler yükleniyor...")
        try:
            df_true = pd.read_csv(true_csv_path)
            df_fake = pd.read_csv(fake_csv_path)
        except Exception as e:
            print(f"Hata: Veri dosyaları okunamadı. Veri setlerinin data/ klasöründe olduğundan emin olun. Detay: {e}")
            return False
            
        # Sınıfları belirleyelim (Real: 1, Fake: 0)
        df_true['class'] = 1
        df_fake['class'] = 0
        
        # Verileri birleştiriyoruz ve karıştırıyoruz
        df = pd.concat([df_true, df_fake], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Hızlı çalışması eğitimde daha iyi olabilir, sadece text kısmını alıyoruz
        # Kaggle datasetinde "text" sütunu ana metni içerir.
        if 'text' not in df.columns:
            print("Veri setinde 'text' isimli bir sütun bulunamadı. Yapıyı kontrol ediniz.")
            return False
            
        # Boş ve NAN olanları çıkaralım
        df.dropna(subset=['text'], inplace=True)
        
        print("Veriler ön işleme aşamasından geçiyor (Bu biraz zaman alabilir)...")
        # Gerçek uygulamalarda binlerce satırda apply yavaş olabilir ama doğruluk için gereklidir.
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        X = df['cleaned_text']
        y = df['class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        print("Metinler Vektör haline (TF-IDF) getiriliyor...")
        self.vectorizer = TfidfVectorizer()
        X_train_idf = self.vectorizer.fit_transform(X_train)
        X_test_idf = self.vectorizer.transform(X_test)
        
        print("Model eğitiliyor (Logistic Regression)...")
        self.model = LogisticRegression()
        self.model.fit(X_train_idf, y_train)
        
        print("\n--- Model Performans Değerlendirmesi ---")
        y_pred = self.model.predict(X_test_idf)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Modeli kaydet!
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
            
        print("Model ve vektörizatör 'models' klasörüne kaydedildi.")
        return True

    def predict(self, text):
        """
        Dışarıdan gelen yeni bir metnin sahte mi gerçek mi olduğunu tahmin eder.
        """
        if self.model is None or self.vectorizer is None:
            raise Exception("Eğitilmiş bir model bulunamadı! Lütfen önce verisetini indirip modeli eğitin.")
            
        cleaned_text = self.preprocessor.clean_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)[0]
        
        prediction_prob = self.model.predict_proba(vectorized_text)[0]
        confidence = max(prediction_prob) * 100
        
        # 1 -> Real, 0 -> Fake
        label = "Real News" if prediction == 1 else "Fake News"
        return label, confidence

if __name__ == "__main__":
    # Konsoldan doğrudan modeli eğitmek isterseniz bu bloğu kullanabilirsiniz.
    # python src/model.py
    fake_model = FakeNewsModel()
    fake_model.train_model("data/True.csv", "data/Fake.csv")
