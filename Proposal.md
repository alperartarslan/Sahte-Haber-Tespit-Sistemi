Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Hedef (Goal)
Bu projede temel amacım, haber metinlerini "Gerçek" veya "Sahte" olarak sınıflandırabilen Yapay Zeka destekli bir sistem geliştirmektir. Doğal Dil İşleme (NLP) teknikleri ve Makine Öğrenimi algoritmaları aracılığıyla metinsel verileri analiz ederek, sistemin aldatıcı dilsel kalıpları tanımlaması hedeflenmektedir. İkincil bir hedef ise, kullanıcıların haber metinlerini gerçek zamanlı olarak kolayca test edebilecekleri (Streamlit kullanarak) kullanıcı dostu bir web arayüzü sağlamaktır.

Veri Setleri (Datasets)
Bu projede kullanılan ana veri seti, Kaggle'dan temin edilen "Fake and Real News Dataset" olacaktır. Bu veri seti, iki ayrı dosyada (True.csv ve Fake.csv) yapılandırılmış, doğru şekilde etiketlenmiş binlerce gerçek ve sahte haber makalesini içermektedir. Haberlerin metin içeriği (content), modelin eğitimi için birincil özellik olarak hizmet edecektir. Ek doğrulama gerekirse, LIAR veri seti gibi alternatif veri setlerinden de yararlanılabilecektir.

Metodoloji (Methodology)
Proje, yapılandırılmamış metni işlemek için sağlam bir NLP boru hattına (pipeline) odaklanacaktır. Metinsel veriler ilk olarak; küçük harfe dönüştürme, noktalama işaretlerini kaldırma, NLTK kütüphanesi kullanılarak durak kelimelerinin (stop-words) ayıklanması ve sözcük kökü bulma (lemmatizasyon) işlemlerini içeren metin ön işlemesine tabi tutulacaktır. Temizlik işleminin ardından metin, TF-IDF (Terim Frekansı-Ters Belge Frekansı) yöntemi kullanılarak sayısal özellik vektörlerine dönüştürülecektir. Bu vektörler daha sonra, özgün ve uydurma haberler arasındaki ayrımı öğrenmek için bir Lojistik Regresyon sınıflandırıcısına (alternatif olarak Random Forest potansiyeli ile) beslenecektir. Modelin performansı; Doğruluk (Accuracy), Kesinlik (Precision), Duyarlılık (Recall) ve F1-Skoru gibi standart metrikler kullanılarak değerlendirilecektir.

Beklenen Sonuç (Expected Outcome)
Projenin beklenen sonucu, modern ve etkileşimli bir Streamlit web uygulamasına entegre edilmiş, yüksek doğruluğa sahip bir metin sınıflandırma modelidir. Kullanıcılar metin girişi yapabilecek veya bir belge yükleyebilecek ve sistem, haberin sahte mi yoksa gerçek mi olduğunu bir güven puanı ile birlikte anında tahmin edecektir. Sonuçlar, NLP ve geleneksel ML modellerinin dezenformasyonla mücadele etmek için nasıl etkili bir şekilde kullanılabileceğini gösterecektir.

GitHub Deposu (GitHub Repository):
https://github.com/KullaniciAdiniz/Fake-News-Detection

Veri Kaynağı Erişimi (Data Source Access):
- Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- LIAR Dataset (Alternatif): https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

Referanslar (References):
- NLTK Dokümantasyonu: https://www.nltk.org/
- Scikit-learn Dokümantasyonu: https://scikit-learn.org/stable/
- Streamlit Dokümantasyonu: https://docs.streamlit.io/
