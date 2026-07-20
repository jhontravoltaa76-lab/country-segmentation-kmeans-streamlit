# Aplikasi Segmentasi Negara (K-Means & PCA)
Aplikasi web interaktif berbasis Streamlit untuk mengelompokkan (clustering) negara-negara di dunia berdasarkan indikator pembangunan dan ekonomi dari World Bank Development Indicators, menggunakan algoritma K-Means dan divisualisasikan dengan PCA (Principal Component Analysis).

Fitur Utama
  -Upload CSV mandiri — pengguna mengunggah dataset World Bank sendiri lewat sidebar.
  -Pemilihan tahun analisis dan jumlah klaster (K) secara interaktif (slider K=2–8).

Pembersihan & imputasi data otomatis:
  -Membersihkan nama kolom dari spasi tersembunyi.
  -Membuang negara yang kehilangan >50% fitur pada tahun terpilih.
  -Membuang kolom indikator yang 100% kosong untuk tahun tersebut.
  -Mengisi (impute) nilai yang hilang dengan median tahun berjalan.
  -Validasi jumlah data minimum terhadap nilai K yang dipilih.



Standarisasi data dengan StandardScaler sebelum clustering.
Clustering K-Means dengan pelabelan dinamis berbasis rangking rata-rata GDP tiap klaster (mis. "Tingkat 1 (Kesejahteraan Tinggi)" saat K=3).
Visualisasi 2D hasil reduksi dimensi PCA menggunakan Plotly (interaktif, hover per negara).
Evaluasi jumlah klaster: grafik Elbow Method (SSE) dan Silhouette Score untuk membantu menentukan K yang optimal.
Pencarian negara untuk melihat segmen & status kualitas data (asli vs. diimputasi).
Profil rata-rata fitur per klaster dengan heatmap tabel untuk interpretasi antar klaster.
Transparansi data: panel status pembersihan data (jumlah negara awal, dibuang, diimputasi) dan tab cuplikan data mentah siap-proses.


Tampilan Aplikasi

Aplikasi terbagi dalam 3 tab utama:
  - Hasil Segmentasi — peta sebaran negara (PCA), pencarian negara, dan daftar negara per segmen.
  - Evaluasi & Profil Klaster — profil rata-rata fitur per klaster, grafik Elbow dan Silhouette Score.
  - Cuplikan Data Mentah — tabel data numerik final yang digunakan model, lengkap dengan penanda data hasil imputasi.


Struktur Project

country-segmentation-kmeans-streamlit/
├── app.py                                  # Kode utama aplikasi Streamlit
├── world_bank_development_indicators.csv   # Dataset contoh (World Bank)
└── README.md

Dataset

Dataset harus berformat CSV dan minimal memiliki kolom berikut:
KolomKeterangancountryNama negaradateTanggal/tahun data (format tanggal)GDP_current_USGDP negara (US$ saat ini) — wajib ada agar clustering bisa jalan

Indikator tambahan yang digunakan (bila tersedia di dataset):
  -life_expectancy_at_birth — angka harapan hidup saat lahir
  -birth_rate — angka kelahiran
  -death_rate — angka kematian
  -access_to_electricity% — akses terhadap listrik
  -government_health_expenditure% — belanja pemerintah untuk kesehatan


Dataset contoh (world_bank_development_indicators.csv) sudah disertakan di repo ini, bersumber dari indikator pembangunan World Bank.

Instalasi
1.  Clone repository
      - bash   git clone https://github.com/jhontravoltaa76-lab/country-segmentation-kmeans-streamlit.git
      - cd country-segmentation-kmeans-streamlit


2.  (Disarankan) Buat virtual environment
      - python -m venv venv
      - aktifkan -> #macOS/Linux source venv/bin/activate   # Windows: venv\Scripts\activate


3. Install semua library yang dibutuhkan
      - bash   pip install streamlit pandas matplotlib scikit-learn numpy plotly


4. Menjalankan Aplikasi
      - streamlit run app.py

Setelah berjalan, buka browser ke alamat yang ditampilkan (biasanya http://localhost:8501), lalu:
  -Upload file CSV dataset World Bank melalui sidebar.
  -Pilih tahun analisis yang diinginkan.
  -Atur jumlah klaster (K) menggunakan slider.
  -Jelajahi hasil segmentasi di ketiga tab yang tersedia.


Metodologi
  -Pemuatan & pembersihan data — membaca CSV, menstandarkan nama kolom, memvalidasi kolom wajib.
  -Penyaringan per tahun — mengambil snapshot data sesuai tahun yang dipilih pengguna.
  -Penanganan data hilang — membuang negara dengan >50% fitur kosong, lalu mengisi sisa nilai kosong dengan median.
  -Standarisasi — menyamakan skala seluruh fitur numerik dengan StandardScaler.
  -Clustering — menjalankan KMeans (dengan n_init=10, random_state=42 untuk hasil konsisten).
  -Pelabelan dinamis — mengurutkan klaster berdasarkan rata-rata GDP agar label ("Tingkat 1", "Tingkat 2", dst.) selalu konsisten dan bermakna.
  -Reduksi dimensi — menggunakan PCA (2 komponen) untuk visualisasi 2D.
  -Evaluasi model — menghitung SSE (Elbow Method) dan Silhouette Score untuk berbagai nilai K sebagai justifikasi pemilihan jumlah klaster.


Teknologi yang Digunakan
  -Streamlit — antarmuka web interaktif
  -Pandas — manipulasi data
  -Scikit-learn — StandardScaler, KMeans, PCA, SimpleImputer, silhouette_score
  -Plotly Express — visualisasi interaktif
  -Matplotlib — grafik Elbow & Silhouette Score
  -NumPy — komputasi numerik
