# Aplikasi Segmentasi Negara (K-Means & PCA)
Aplikasi web interaktif berbasis Streamlit untuk mengelompokkan (clustering) negara-negara di dunia berdasarkan indikator pembangunan dan ekonomi dari World Bank Development Indicators, menggunakan algoritma K-Means dan divisualisasikan dengan PCA (Principal Component Analysis).

## Fitur Utama
1. Upload CSV mandiri — pengguna mengunggah dataset World Bank sendiri lewat sidebar.
2. Pemilihan tahun analisis dan jumlah klaster (K) secara interaktif (slider K=2–8).
3. Pembersihan & imputasi data otomatis:
    -Membersihkan nama kolom dari spasi tersembunyi.
    -Membuang negara yang kehilangan >50% fitur pada tahun terpilih.
    -Membuang kolom indikator yang 100% kosong untuk tahun tersebut.
    -Mengisi (impute) nilai yang hilang dengan median tahun berjalan.
    -Validasi jumlah data minimum terhadap nilai K yang dipilih.
4. Standarisasi data dengan StandardScaler sebelum clustering.

## Tampilan Aplikasi
Aplikasi terbagi dalam 3 tab utama:
  - Hasil Segmentasi — peta sebaran negara (PCA), pencarian negara, dan daftar negara per segmen.
  - Evaluasi & Profil Klaster — profil rata-rata fitur per klaster, grafik Elbow dan Silhouette Score.
  - Cuplikan Data Mentah — tabel data numerik final yang digunakan model, lengkap dengan penanda data hasil imputasi.


## Struktur Project

country-segmentation-kmeans-streamlit/
├── app.py                                  # Kode utama aplikasi Streamlit
├── world_bank_development_indicators.csv   # Dataset contoh (World Bank)
└── README.md

## Dataset
Dataset harus berformat CSV dan minimal memiliki kolom berikut:
KolomKeterangancountryNama negaradateTanggal/tahun data (format tanggal)GDP_current_USGDP negara (US$ saat ini) — wajib ada agar clustering bisa jalan

Indikator tambahan yang digunakan (bila tersedia di dataset):
  -life_expectancy_at_birth — angka harapan hidup saat lahir
  -birth_rate — angka kelahiran
  -death_rate — angka kematian
  -access_to_electricity% — akses terhadap listrik
  -government_health_expenditure% — belanja pemerintah untuk kesehatan


Dataset contoh (world_bank_development_indicators.csv) sudah disertakan di repo ini, bersumber dari indikator pembangunan World Bank.

## Instalasi
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
  - Upload file CSV dataset World Bank melalui sidebar.
  - Pilih tahun analisis yang diinginkan.
  - Atur jumlah klaster (K) menggunakan slider.
  - Jelajahi hasil segmentasi di ketiga tab yang tersedia.

## Teknologi yang Digunakan
  - Streamlit
  - Pandas
  - Scikit-learn
  - Plotly Express
  - Matplotlib
  - NumPy
