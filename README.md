# Klasifikasi Golongan Suara dalam Paduan Suara dengan Pendekatan Fine Tuning Pretrained Model Wav2Vec2
**Stefanus - 23523036**

## Abstrak
Menentukan golongan suara dalam paduan suara adalah aspek yang sangat penting dalam paduan suara. Penentuan golongan suara atau yang biasa disebut ambitus memerlukan pemahaman mendalam tentang musik. Fakta ini menjadikan pembangunan model klasifikasi golongan suara dalam paduan suara sebagai eksperimen multidisiplin. Selain tinjauan teknis, tinjauan musikalitas yang mendalam juga diperlukan. Oleh karena itu, menentukan jenis vokal dalam paduan suara tidak dapat dilakukan hanya berdasarkan aturan mengenai jangkauan suara seseorang. Dalam penelitian ini, data dikumpulkan dari anggota Paduan Suara Mahasiswa ITB (PSM-ITB) dengan proses pelabelan dilakukan oleh tim pelatih vokal PSM-ITB.

## Latar Belakang
Musik adalah bahasa universal yang dapat dipahami tanpa mempedulikan bahasa, budaya, atau selera. Musik dapat menyampaikan pesan kepada pendengarnya dan penikmatnya. Paduan suara adalah salah satu jenis musik tertua dengan bukti keberadaannya yang dapat ditelusuri kembali ke Yunani Kuno. Polifoni adalah konsep dalam paduan suara di mana pada waktu tertentu sumber musik tidak hanya satu, tetapi beberapa sumber musik yang secara bersamaan membunyikan nada masing-masing menciptakan harmoni yang indah. Dalam paduan suara, polifoni ini umumnya diwujudkan dengan empat golongan suara utama yaitu Sopran, Alto, Tenor, dan Bass yang biasa disingkat SATB.

Memilih golongan suara yang salah dapat sangat merugikan baik secara artistik maupun kesehatan pita suara penyanyi. Ketika seorang penyanyi memaksakan nada yang terlalu tinggi atau terlalu rendah untuk jangkauannya, hal ini memberi tekanan besar pada pita suara yang berpotensi menyebabkan kerusakan permanen. Oleh karena itu, sangat penting untuk menentukan golongan suara yang tepat berdasarkan karakteristik suara penyanyi.

Solusi yang diajukan dalam penelitian ini adalah penggunaan Wav2Vec sebagai model baseline yang dikembangkan oleh Facebook. Wav2Vec sendiri pada umumnya digunakan sebagai model Automatic Speech Recognition. Namun karena Wav2Vec sendiri tidak menyediakan tokenizer, Wav2Vec dapat digunakan sebagai baseline dari model klasifikasi.

## Studi Literatur

### Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
Paper ini merupakan paper utama dari Wav2Vec 2.0 buatan Facebook AI. Paper ini memperkenalkan Wav2Vec 2.0, sebuah framework untuk self-supervised learning representasi suara. Model ini menggunakan Convolutional Neural Network untuk meng-encode audio mentah dan kemudian melakukan masking sebagian dari representasi suara laten untuk menyelesaikan task-task tertentu. Wav2Vec 2.0 menunjukkan hasil yang sangat baik dalam pengenalan suara, mencapai Word Error Rate (WER) sebesar 1.8/3.3 pada Librispeech dengan menggunakan seluruh data berlabel. Dengan menggunakan 10 menit data berlabel, model ini masih mampu mencapai WER 4.8/8.2. Wav2Vec 2.0 dapat mencapai kinerja tinggi dengan sangat sedikit data berlabel, menunjukkan efisiensi dalam penggunaan data tidak berlabel untuk pretraining​. Namun sayangnya, model ini merupakan model transformer yang besar dan metode pretraining memerlukan sumber daya komputasi yang tinggi.

Berikut merupakan blog penjelasan lebih mendetail mengenai Wav2Vec:  
[Wav2Vec 2.0 Blog](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)

Kemudian berikut merupakan paper-paper yang menggunakan Wav2Vec untuk task-task tertentu, terutama klasifikasi:

### Exploring Wav2Vec 2.0 Fine-Tuning for Improved Speech Emotion Recognition
Paper ini mengeksplorasi metode untuk meningkatkan Speech Emotion Recognition (SER) menggunakan model Wav2Vec 2.0. Model ini dilatih untuk mengenali emosi dalam ucapan dengan memprediksi unit suara dari bagian audio yang disamarkan. Studi ini membandingkan metode fine-tuning dasar seperti vanilla fine-tuning (V-FT) dan task adaptive pretraining (TAPT) serta memperkenalkan metode baru P-TAPT.

Metode P-TAPT menunjukkan hasil terbaik dalam peningkatan kinerja SER terutama dalam kondisi low resource. Metode P-TAPT secara signifikan meningkatkan akurasi pengenalan emosi dibandingkan dengan model sebelumnya. Metode P-TAPT (Pre-trained Task Adaptive Pretraining) adalah pendekatan yang memperluas Task Adaptive Pretraining (TAPT). TAPT menyesuaikan model dengan tugas spesifik sebelum fine-tuning. P-TAPT memodifikasi TAPT untuk menghasilkan representasi emosi yang lebih kontekstual.

Dengan menyesuaikan model pada data terkait tugas emosi sebelum pelatihan akhir, P-TAPT meningkatkan kemampuan model untuk mengenali emosi dalam ucapan bahkan dalam kondisi data berlabel yang terbatas. Metode ini telah menunjukkan peningkatan signifikan dalam kinerja pengenalan emosi ucapan dibandingkan metode sebelumnya.

Berikut merupakan repository yang dikembangkan:  
[FT-w2v2-ser Repository](https://github.com/b04901014/FT-w2v2-ser)

### Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings
Paper ini mengeksplorasi penggunaan model Wav2Vec 2.0 untuk pengenalan emosi dari ucapan. Metode yang digunakan melibatkan transfer learning di mana fitur diekstrak dari model Wav2Vec 2.0 yang sudah dilatih sebelumnya dan kemudian dimodelkan menggunakan neural network sederhana. Studi ini menunjukkan bahwa menggunakan representasi dari Wav2Vec 2.0 menghasilkan kinerja yang lebih baik dalam pengenalan emosi dibandingkan dengan metode tradisional. Model yang menggabungkan output dari beberapa lapisan Wav2Vec 2.0 dengan bobot yang dapat dilatih memberikan hasil yang superior pada database emotion recognition seperti IEMOCAP dan RAVDESS. Menggunakan Wav2Vec 2.0 memungkinkan pelatihan yang lebih efektif bahkan dengan data yang terbatas.

Berikut merupakan repository yang dikembangkan:  
[ser-with-w2v2 Repository](https://github.com/habla-liaa/ser-with-w2v2)

### Speech-based Age and Gender Prediction with Transformers
Paper ini mengeksplorasi penggunaan model Wav2Vec 2.0 untuk prediksi usia dan gender berdasarkan suara. Penelitian ini mengkurasi beberapa dataset publik dan melakukan eksperimen untuk menguji model pada berbagai dataset tersebut. Hasil menunjukkan bahwa model dapat memprediksi usia dengan MAE antara 5-6 tahun dan mencapai akurasi hingga 90% untuk prediksi gender. Sama seperti paper-paper lainnya, metode fine tuning memungkinkan pelatihan yang efektif dengan data yang terbatas.

Berikut merupakan model yang dikembangkan:  
[wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)

### Classification of Vocal Type in Choir Using Convolutional Recurrent Neural Network (CRNN)
Paper ini merupakan paper karya tulisan saya sendiri bersama dosen (Dessi Puji Lestari). Pada paper ini dilakukan eksplorasi terhadap fitur-fitur yang digunakan dalam pengklasifikasian golongan suara. Penelitian ini menggunakan model Convolutional Recurrent Neural Network sederhana untuk memverifikasi keabsahan fitur-fitur yang diekstrak dalam pengklasifikasian golongan suara yang mana berhasil diekstrak oleh layer convolutional. Oleh sebab itu, Wav2Vec 2.0 yang juga menggunakan layer convolutional sebagai feature extractor dipercaya dapat digunakan sebagai baseline finetuning model.

## Dataset
Sama seperti rancangan solusi pembelajaran mesin umumnya, tahap awal yang dilakukan dalam pengembangan model pembelajaran mesin pengklasifikasian golongan suara adalah tahap pengumpulan dataset. Dataset yang dikumpulkan bersumber anggota Paduan Suara Mahasiswa ITB (PSM-ITB). Dalam pembangkitan dataset ini, perekaman dilakukan dengan merekam nada C3 dan C4 bagi pria serta C4 dan C5 bagi wanita.

Pemilihan nada C baik C3, C4, maupun C5 adalah karena berdasarkan vocal range yang menjadi acuan, nada tersebut merupakan nada yang secara teoritis dapat dijangkau semua jenis golongan suara. Nada direkam selama paling singkat 1 detik untuk tiap nadanya. Perekaman tidak menggunakan ruang studio melainkan sebatas diberi instruksi untuk merekam menggunakan device masing-masing di ruangan yang senyap.

Data audio yang sudah dikumpulkan pertama diseragamkan melalui tahapan-tahapan preprocessing, yakni menggunakan aplikasi editor audio Audacity. Preprocessing yang dilakukan adalah melakukan cutting sedemikian rupa sehingga nada yang diambil memiliki panjang yang sama. Setelah itu, bagian perpindahan nada dihapuskan sedemikian rupa sehingga bagian perpindahan nada nantinya tidak masuk ke dalam perhitungan. Selain itu, diberikan zero-padding sedemikian rupa sehingga waktu mulai dan berakhir tiap nada berada di titik yang sama. Seluruh audio dikonversi ke dalam bentuk .wav dengan sampling rate senilai 16000 hz (16 khz).

Jumlah data yang digunakan adalah 29 data untuk setiap golongan suara yang berarti 29\*4=116 data. Dilakukan pemisahan train-test split dengan rasio 80:20 menggunakan teorema Pareto sebagai acuan. Diberikan juga stratify guna menyeimbangkan data dan m

encegah overfitting.

## Metode
Metode yang digunakan dalam penelitian ini dimulai dengan pembuatan dataset. Dataset yang sudah diolah sebelumnya pertama-tama disimpan waveform yang didapatkan dari library torchaudio sebelumnya. Data ini disimpan dalam bentuk tuple dengan format `{"input_values": input_values, "label": label}` yang input_values-nya berupa representasi waveform dalam bentuk array dan label berupa label yang telah di-encode sebelumnya. Dataset ini kemudian di-wrap dengan library Datasets guna mengubah bentuknya menjadi DatasetDict.

Kemudian dibuat trainer menggunakan library transformer dan menggunakan pretrained model “facebook/wav2vec2-base”.  Kemudian dilakukan finetuning dengan dataset train. Yang lalu diakhiri dengan inferensi untuk melakukan evaluasi kinerja dari model.

## Eksperimen
Eksperimen dilakukan dengan infrastruktur berupa:
```
- GPU Nvidia RTX A5000 16 GB RAM DDR5
- CPU Intel Xeon Silver 4208 CPU @2.10GHz
- RAM 128 GB DDR4
```

Menggunakan infrastruktur ini, untuk menyelesaikan 33 epoch dibutuhkan waktu selama 104 detik (1 menit 44 detik).

Training arguments yang digunakan disesuaikan dari tutorial finetuning dari pranala berikut:  
[Fine-Tuning Wav2Vec2](https://medium.com/@vi.ai_/fine-tuning-wav2vec2-on-your-google-colab-take-a-deep-dive-into-advanced-audio-classification-5db65b7b3bf0)

Dilakukan perubahan di beberapa argumen seperti learning rate yang dilakukan guna menyesuaikan jumlah dataset yang relatif sedikit. Hal ini dilakukan dengan rule-of-thumb yaitu memperhatikan learning curve dari training. Ditambahkan juga parameter report-to guna mengabaikan penggunaan API untuk post data hasil training ke komunitas.

```python
training_args = TrainingArguments(
    output_dir="vocal_classification",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=33,
    warmup_ratio=0.1,
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=[] 
)
```

Didapatkan hasil dari testing data sebagai berikut:

### Confusion Matrix:
```
|       | Alto | Bass | Sopran | Tenor |
|-------|------|------|--------|-------|
| Alto  | 4    | 0    | 2      | 0     |
| Bass  | 0    | 5    | 0      | 1     |
| Sopran| 1    | 1    | 4      | 0     |
| Tenor | 0    | 2    | 0      | 4     |
```

### Classification Report:
```
              precision    recall  f1-score   support
        alto       0.80      0.67      0.73         6
        bass       0.62      0.83      0.71         6
      sopran       0.67      0.67      0.67         6
       tenor       0.80      0.67      0.73         6
    accuracy                           0.71        24
   macro avg       0.72      0.71      0.71        24
weighted avg       0.72      0.71      0.71        24
```

Berikut merupakan hasil inferensi antara true label dan predict label:
```
| Filename          | True Label | Predicted Label |
|-------------------|------------|-----------------|
| validation_0.wav  | tenor      | bass            |
| validation_1.wav  | alto       | alto            |
| validation_2.wav  | bass       | bass            |
| validation_3.wav  | alto       | sopran          |
| validation_4.wav  | sopran     | sopran          |
| validation_5.wav  | tenor      | tenor           |
| validation_6.wav  | bass       | bass            |
| validation_7.wav  | tenor      | tenor           |
| validation_8.wav  | bass       | bass            |
| validation_9.wav  | tenor      | bass            |
| validation_10.wav | tenor      | tenor           |
| validation_11.wav | bass       | bass            |
| validation_12.wav | tenor      | tenor           |
| validation_13.wav | alto       | alto            |
| validation_14.wav | alto       | sopran          |
| validation_15.wav | alto       | alto            |
| validation_16.wav | sopran     | sopran          |
| validation_17.wav | sopran     | sopran          |
| validation_18.wav | alto       | alto            |
| validation_19.wav | sopran     | alto            |
| validation_20.wav | bass       | tenor           |
| validation_21.wav | bass       | bass            |
| validation_22.wav | sopran     | sopran          |
| validation_23.wav | sopran     | bass            |
```
Hasil yang didapatkan relatif baik mengingat keterbatasan data train maupun validation. Didapatkan akurasi di angka 71% yang mana cukup baik. Namun bila dilihat secara kualitatif, mayoritas kesalahan terjadi dengan tidak begitu jauh yakni tenor diidentifikasi sebagai bass dan sebaliknya maupun sopran diidentifikasi sebagai alto dan sebaliknya. Anomali hanya terjadi di validation_23 yang setelah dilakukan cherrypicking ternyata merupakan audio kosong. Hal ini menjelaskan bahwa secara kualitatif hasil yang didapatkan sudah cukup baik.

IPYNB dan dataset yang dikerjakan dalam penelitian ini dapat diakses pada repositori GitHub: [stefanus-lamlo/IF5281](https://github.com/stefanus-lamlo/IF5281)

## Hasil
Hasil yang didapatkan dalam eksperimen sederhana ini cukup baik namun belum bisa mengungguli klasifikasi tradisional menggunakan feature extraction yang di-cherrypick yakni Mel-Frequent Cepstral Coefficient. Hal ini disebabkan Wav2Vec2 bukanlah model monopurpose melainkan model multipurpose yang dapat digunakan di hampir semua aspek speech processing. Hal yang juga mempengaruhi hal ini adalah data train dan test yang sangat sedikit tidak cukup untuk melakukan finetuning pada model sebesar Wav2Vec2 secara efektif. Kedepannya, bila ingin menggunakan model pretrained, harus dicoba untuk melakukan freezing pada layer-layer tertentu dan dilakukan modifikasi layer sehingga dapat melakukan feature extraction yang lebih sesuai seperti dijelaskan pada paper “Exploring Wav2Vec 2.0 Fine-Tuning for Improved Speech Emotion Recognition”.
