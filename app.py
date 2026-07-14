import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.express as px

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Segmentasi Negara", page_icon="🌍", layout="wide")

st.title("Aplikasi Segmentasi Negara (K-Means & PCA) 🌍")
st.markdown(
    "Analisis Klaster: Mengelompokkan Negara berdasarkan indikator pembangunan dan ekonomi dari Bank Dunia."
)
st.markdown("---")


# --- 1. FUNGSI UNTUK MEMUAT DATA MENTAH (DI-CACHE) ---
@st.cache_data
def load_raw_data(file_csv):
    try:
        df = pd.read_csv(file_csv)
    except Exception as e:
        return None, f"Error membaca file: {e}"

    # Pembersihan nama kolom (Mengantisipasi spasi tersembunyi - Issue #6)
    df.columns = df.columns.str.strip()

    required_cols = ["country", "date", "GDP_current_US"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return (
            None,
            f"Format file tidak valid. Kolom wajib yang hilang: {', '.join(missing_cols)}",
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    return df, None


# --- 2. FUNGSI UNTUK MENYIAPKAN & IMPUTASI DATA (DI-CACHE) ---
@st.cache_data
def prepare_data(df, selected_year, k_terpilih):
    kolom_pilihan = [
        "country",
        "year",
        "GDP_current_US",
        "life_expectancy_at_birth",
        "birth_rate",
        "death_rate",
        "access_to_electricity%",
        "government_health_expenditure%",
    ]
    kolom_tersedia = [col for col in kolom_pilihan if col in df.columns]
    df_selected = df[kolom_tersedia]

    df_snapshot = df_selected[df_selected["year"] == selected_year].copy()
    initial_row_count = len(df_snapshot)

    empty_result = {
        "df_numeric": None,
        "data_scaled": None,
        "initial_rows": initial_row_count,
        "final_rows": 0,
        "imputed_countries": [],
        "dropped_cols": [],
        "imputation_ratio": 0.0,
    }

    if initial_row_count == 0:
        return empty_result

    df_snapshot = df_snapshot.set_index("country")
    df_numeric = df_snapshot.drop(columns=["year"])
    fitur_awal = list(df_numeric.columns)

    # 2A. Toleransi: Hapus negara yang kehilangan > 50% fitur (Issue #7 - Hybrid Thresholding)
    threshold = len(df_numeric.columns) / 2
    df_numeric = df_numeric.dropna(thresh=threshold)

    # Hapus kolom yang 100% kosong (NaN) agar SimpleImputer tidak error karena mismatch shape
    df_numeric = df_numeric.dropna(axis=1, how="all")

    # Deteksi kolom yang hilang total untuk tahun ini (indikator tidak tersedia sama sekali)
    dropped_cols = [c for c in fitur_awal if c not in df_numeric.columns]

    final_row_count = len(df_numeric)

    # Validasi minimum data terhadap K (Issue #5) & pastikan GDP tidak hilang
    if final_row_count <= k_terpilih or "GDP_current_US" not in df_numeric.columns:
        empty_result["dropped_cols"] = dropped_cols
        return empty_result

    # 2B. Deteksi negara mana saja yang perlu diimputasi (Untuk Transparansi UI)
    countries_with_missing = df_numeric[df_numeric.isnull().any(axis=1)].index.tolist()
    imputation_ratio = (
        len(countries_with_missing) / final_row_count if final_row_count else 0.0
    )

    # 2C. Imputasi (Isi sisa nilai kosong dengan Median tahun tersebut, tahan terhadap outlier)
    imputer = SimpleImputer(strategy="median")
    data_imputed = imputer.fit_transform(df_numeric)

    # Kembalikan ke format DataFrame
    df_numeric_imputed = pd.DataFrame(
        data_imputed, columns=df_numeric.columns, index=df_numeric.index
    )

    # Standarisasi Data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_numeric_imputed)

    return {
        "df_numeric": df_numeric_imputed,
        "data_scaled": data_scaled,
        "initial_rows": initial_row_count,
        "final_rows": final_row_count,
        "imputed_countries": countries_with_missing,
        "dropped_cols": dropped_cols,
        "imputation_ratio": imputation_ratio,
    }


# --- 3. FUNGSI CACHE UNTUK ELBOW METHOD (Issue #2) ---
@st.cache_data
def calculate_elbow_sse(data_scaled, max_k=10):
    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans_elbow = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans_elbow.fit(data_scaled)
        sse.append(kmeans_elbow.inertia_)
    return list(k_range), sse


# --- 3B. FUNGSI CACHE UNTUK SILHOUETTE SCORE (Validasi tambahan selain Elbow) ---
@st.cache_data
def calculate_silhouette_scores(data_scaled, max_k=10):
    n_samples = len(data_scaled)
    scores = {}
    for k in range(2, max_k + 1):
        if k >= n_samples:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(data_scaled)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(data_scaled, labels)
    return scores


# --- 4. FITUR UPLOAD & EKSEKUSI UTAMA ---
st.sidebar.header("📁 Unggah Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV World Bank Development", type=["csv"]
)
st.sidebar.info(
    "Gunakan dataset indikator Bank Dunia yang berisi kolom 'country', 'date', dan 'GDP_current_US'."
)

if uploaded_file is None:
    st.info(
        "👈 Silakan upload file CSV terlebih dahulu melalui sidebar di sebelah kiri untuk memulai analisis."
    )
else:
    df_raw, error_msg = load_raw_data(uploaded_file)

    if error_msg:
        st.error(error_msg)
    else:
        available_years = sorted(
            df_raw["year"].dropna().unique().astype(int).tolist(), reverse=True
        )

        st.sidebar.markdown("---")
        st.sidebar.header("📅 Pengaturan Analisis")

        default_idx = available_years.index(2020) if 2020 in available_years else 0
        selected_year = st.sidebar.selectbox(
            "Pilih Tahun Analisis:", available_years, index=default_idx
        )

        # Opsi interaktif untuk K (Otomatis didukung oleh Pelabelan Dinamis Issue #1)
        K_TERPILIH = st.sidebar.slider(
            "Jumlah Klaster (Rekomendasi K3):", min_value=2, max_value=8, value=3
        )

        # Menyiapkan Data
        hasil = prepare_data(df_raw, selected_year, K_TERPILIH)
        df_numeric = hasil["df_numeric"]
        data_scaled = hasil["data_scaled"]
        initial_rows = hasil["initial_rows"]
        final_rows = hasil["final_rows"]
        imputed_countries = hasil["imputed_countries"]
        dropped_cols = hasil["dropped_cols"]
        imputation_ratio = hasil["imputation_ratio"]

        # Penanganan error jumlah data minimum (Issue #5)
        if df_numeric is None or final_rows <= K_TERPILIH:
            st.warning(
                f"⚠️ Maaf, data untuk tahun **{selected_year}** terlalu sedikit (tersisa {final_rows} negara setelah difilter). Data harus lebih banyak dari jumlah klaster (K={K_TERPILIH}). Silakan pilih tahun lain."
            )
            if dropped_cols:
                st.error(
                    f"🚫 Selain itu, indikator berikut tidak tersedia sama sekali di tahun {selected_year}: **{', '.join(dropped_cols)}**"
                )
        else:
            dropped_rows = initial_rows - final_rows

            # --- PERINGATAN: Indikator yang hilang total di tahun ini (Temuan baru) ---
            if dropped_cols:
                st.warning(
                    f"⚠️ **Indikator tidak tersedia untuk tahun {selected_year}:** "
                    f"{', '.join(dropped_cols)}. Analisis tahun ini hanya menggunakan "
                    f"{len(df_numeric.columns)} dari {len(df_numeric.columns) + len(dropped_cols)} indikator awal. "
                    "Akibatnya, hasil klaster tahun ini **tidak sepenuhnya sebanding** dengan tahun-tahun "
                    "yang datanya lebih lengkap."
                )

            # --- PERINGATAN: Rasio imputasi tinggi (Temuan baru) ---
            if imputation_ratio > 0.3:
                st.warning(
                    f"⚠️ **Rasio imputasi tinggi:** {imputation_ratio:.0%} dari negara pada tahun "
                    f"{selected_year} memiliki setidaknya satu nilai yang diisi otomatis (median tahun ini), "
                    "bukan data asli. Interpretasikan hasil segmentasi tahun ini dengan hati-hati."
                )

            # Notifikasi Transparansi Pembersihan Data (Issue #7 UI Tracker)
            with st.expander(
                f"📊 Status Pembersihan Data (Siap: {final_rows} Negara)",
                expanded=False,
            ):
                st.write(f"- Total Negara Awal: **{initial_rows}**")
                st.write(
                    f"- Negara Dibuang (Kehilangan >50% indikator): **{dropped_rows}**"
                )
                st.write(
                    f"- Negara Diimputasi (Diisi nilai median tahun ini): "
                    f"**{len(imputed_countries)} ({imputation_ratio:.0%})**"
                )
                if dropped_cols:
                    st.write(
                        f"- Indikator dikecualikan total: **{', '.join(dropped_cols)}**"
                    )
                if imputed_countries:
                    st.caption(f"Negara hasil imputasi: {', '.join(imputed_countries)}")

            # --- PROSES CLUSTERING ---
            kmeans = KMeans(n_clusters=K_TERPILIH, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(data_scaled)
            df_numeric["cluster"] = clusters

            # Labeling Dinamis Berbasis Rank PDB (Issue #1 & #4)
            cluster_means = (
                df_numeric.groupby("cluster")["GDP_current_US"]
                .mean()
                .sort_values(ascending=False)
            )
            mapping = {}
            for rank, (cluster_id, _) in enumerate(cluster_means.items()):
                base_label = f"Tingkat {rank + 1}"
                # Tambahan label spesifik hanya jika K==3
                if K_TERPILIH == 3:
                    if rank == 0:
                        base_label += " (Kesejahteraan Tinggi)"
                    elif rank == 1:
                        base_label += " (Kesejahteraan Menengah)"
                    elif rank == 2:
                        base_label += " (Kesejahteraan Rendah)"
                mapping[cluster_id] = base_label

            df_numeric["Segmentasi"] = df_numeric["cluster"].map(mapping)

            # PCA Deterministic (Issue #3)
            pca = PCA(n_components=2, random_state=42)
            data_pca = pca.fit_transform(data_scaled)
            df_pca = pd.DataFrame(
                data_pca, columns=["PC1", "PC2"], index=df_numeric.index
            )
            df_pca["Segmentasi"] = df_numeric["Segmentasi"]

            # --- TATA LETAK UI TABS ---
            tab1, tab2, tab3 = st.tabs(
                [
                    "📊 Hasil Segmentasi",
                    "📈 Evaluasi & Profil Klaster",
                    "🗃️ Cuplikan Data Mentah",
                ]
            )

            with tab1:
                st.subheader(f"Peta Sebaran Negara (2D PCA) - Tahun {selected_year}")
                df_plot = df_pca.reset_index()

                # Mengurutkan legend berdasarkan Tingkat 1, 2, 3...
                unique_labels = sorted(
                    df_plot["Segmentasi"].unique(),
                    key=lambda s: int(s.split()[1]) if s.split()[1].isdigit() else 0,
                )
                category_orders = {"Segmentasi": unique_labels}

                fig_cluster = px.scatter(
                    df_plot,
                    x="PC1",
                    y="PC2",
                    color="Segmentasi",
                    hover_name="country",
                    title=f"Peta Klaster Ekonomi ({selected_year})",
                    category_orders=category_orders,
                    opacity=0.8,
                )
                fig_cluster.update_traces(
                    marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey"))
                )
                fig_cluster.update_layout(
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    )
                )
                st.plotly_chart(fig_cluster, use_container_width=True)

                st.subheader("Pencarian & Daftar Negara")

                semua_negara = sorted(df_numeric.index.tolist())
                cari_negara = st.selectbox(
                    "🔍 Cari nama negara untuk mengetahui segmennya:",
                    ["-- Pilih Negara --"] + semua_negara,
                )

                if cari_negara != "-- Pilih Negara --":
                    segmen_ditemukan = df_numeric.loc[cari_negara, "Segmentasi"]
                    is_imputed = (
                        "⚠️ Sebagian data diimputasi (Median)"
                        if cari_negara in imputed_countries
                        else "✅ Data asli lengkap"
                    )

                    st.info(
                        f"Berdasarkan analisis klaster tahun **{selected_year}**, **{cari_negara}** masuk ke kategori:"
                    )
                    st.markdown(
                        f"<h3 style='text-align: center; color: #3B82F6;'>{segmen_ditemukan}</h3>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Status Kualitas Data: {is_imputed}")
                    st.divider()

                # Tampilkan expander per segmen secara dinamis (grid rapi, maks 4 kolom per baris)
                n_cols = min(K_TERPILIH, 4)
                for row_start in range(0, len(unique_labels), n_cols):
                    row_labels = unique_labels[row_start : row_start + n_cols]
                    cols = st.columns(n_cols)
                    for col, segment_name in zip(cols, row_labels):
                        with col:
                            with st.expander(f"📁 {segment_name}"):
                                negara_di_cluster = df_numeric[
                                    df_numeric["Segmentasi"] == segment_name
                                ].index.tolist()
                                st.markdown(", ".join(negara_di_cluster))

            with tab2:
                st.subheader("Profil Rata-rata Fitur per Klaster")
                st.markdown(
                    "Tabel ini menampilkan **nilai rata-rata asli (bukan skala)** dari tiap indikator untuk masing-masing klaster, membantu Anda menginterpretasikan apa yang membedakan satu klaster dengan yang lain."
                )

                # Menghitung Profil Cluster
                profile_df = (
                    df_numeric.drop(columns=["cluster"]).groupby("Segmentasi").mean()
                )
                profile_df = profile_df.loc[
                    [l for l in unique_labels if l in profile_df.index]
                ]
                st.dataframe(
                    profile_df.style.background_gradient(cmap="Greens", axis=0)
                )

                st.divider()
                st.subheader("Evaluasi Model: Justifikasi Jumlah Klaster (K)")

                col_elb1, col_elb2 = st.columns(2)

                with col_elb1:
                    st.markdown("**Metode Elbow (SSE)**")
                    k_range, sse = calculate_elbow_sse(data_scaled)

                    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
                    ax_elbow.plot(
                        k_range, sse, marker="o", linestyle="-", color="#636EFA"
                    )

                    if K_TERPILIH <= max(k_range):
                        ax_elbow.annotate(
                            f"K Saat Ini ({K_TERPILIH})",
                            xy=(K_TERPILIH, sse[K_TERPILIH - 1]),
                            xytext=(
                                K_TERPILIH + 0.5,
                                sse[K_TERPILIH - 1] + (sse[0] * 0.1),
                            ),
                            arrowprops=dict(
                                facecolor="red", shrink=0.05, width=2, headwidth=8
                            ),
                            fontsize=9,
                            color="red",
                            fontweight="bold",
                        )

                    ax_elbow.set_title("SSE vs Jumlah K")
                    ax_elbow.set_xlabel("Jumlah Cluster (K)")
                    ax_elbow.set_ylabel("Inertia (SSE)")
                    ax_elbow.grid(True, linestyle="--", alpha=0.7)
                    st.pyplot(fig_elbow)

                with col_elb2:
                    st.markdown("**Silhouette Score (semakin tinggi semakin baik)**")
                    silhouette_scores = calculate_silhouette_scores(data_scaled)

                    if silhouette_scores:
                        fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
                        ks = list(silhouette_scores.keys())
                        vals = list(silhouette_scores.values())
                        ax_sil.plot(
                            ks, vals, marker="o", linestyle="-", color="#10B981"
                        )

                        if K_TERPILIH in silhouette_scores:
                            ax_sil.annotate(
                                f"K Saat Ini ({K_TERPILIH})",
                                xy=(K_TERPILIH, silhouette_scores[K_TERPILIH]),
                                xytext=(
                                    K_TERPILIH + 0.5,
                                    silhouette_scores[K_TERPILIH] - 0.05,
                                ),
                                arrowprops=dict(
                                    facecolor="red", shrink=0.05, width=2, headwidth=8
                                ),
                                fontsize=9,
                                color="red",
                                fontweight="bold",
                            )
                            st.caption(
                                f"Skor silhouette untuk K={K_TERPILIH}: "
                                f"**{silhouette_scores[K_TERPILIH]:.3f}** "
                                "(rentang -1 s.d. 1, makin dekat 1 makin baik pemisahan klasternya)."
                            )

                        ax_sil.set_title("Silhouette Score vs Jumlah K")
                        ax_sil.set_xlabel("Jumlah Cluster (K)")
                        ax_sil.set_ylabel("Silhouette Score")
                        ax_sil.grid(True, linestyle="--", alpha=0.7)
                        st.pyplot(fig_sil)
                    else:
                        st.info("Data tidak cukup untuk menghitung silhouette score.")

            with tab3:
                st.subheader("Data Siap Proses")
                st.markdown(
                    f"Tabel murni fitur numerik tahun **{selected_year}** yang diumpankan ke model K-Means."
                )
                if dropped_cols:
                    st.caption(
                        f"Catatan: indikator {', '.join(dropped_cols)} tidak ditampilkan karena "
                        f"tidak tersedia sama sekali untuk tahun {selected_year}."
                    )

                # Tambahkan kolom indikator transparansi UI
                df_display = df_numeric.drop(
                    columns=["cluster", "Segmentasi"], errors="ignore"
                ).copy()
                df_display["Diimputasi?"] = df_display.index.isin(imputed_countries)
                st.dataframe(df_display)
