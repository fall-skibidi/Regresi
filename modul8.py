## Aplikasi Penyelesaian SPL (Regresi Linear - Normal Equation) dengan Streamlit
## Jalankan dengan: streamlit run app_regresi.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Fungsi-fungsi bantu ---
def baca_file(file):
    """Membaca file CSV atau Excel menjadi DataFrame."""
    if file.name.lower().endswith(".csv"):
        # sep=None -> pandas mendeteksi otomatis pemisah (koma atau titik koma)
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file)

def bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep):
    """Membentuk matriks desain X dan vektor target y dari DataFrame."""
    data = df[kolom_fitur + [kolom_target]].apply(pd.to_numeric, errors="raise")
    if data.isnull().any().any():
        raise ValueError("Terdapat sel kosong/NaN pada kolom yang dipilih.")
    
    X = data[kolom_fitur].to_numpy(dtype=float)
    y = data[kolom_target].to_numpy(dtype=float)
    nama_w = [f"w{i+1}" for i in range(len(kolom_fitur))]
    
    if pakai_intersep:
        X = np.column_stack([np.ones(len(X)), X])  # tambah kolom 1 di kiri
        nama_w = ["w0"] + nama_w
        
    return X, y, nama_w

def selesaikan_normal_equation(X, y):
    """Menyelesaikan (X^T X) w = X^T y dengan solver."""
    XtX = X.T @ X  # perkalian matriks
    Xty = X.T @ y
    
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError(
            "Matriks X^T X singular (tidak punya invers). Kemungkinan ada fitur yang "
            "saling bergantung linear atau jumlah sampel kurang dari jumlah parameter."
        )
        
    w = np.linalg.solve(XtX, Xty)  # solver SPL
    return XtX, Xty, w

# --- Tampilan Utama ---
st.set_page_config(page_title="SPL Regresi Linear", layout="wide")
st.title("Penyelesaian SPL dengan Streamlit")
st.caption("Regresi linear berganda melalui Normal Equation: $(X^T X)w = X^T y$")

# Modifikasi: Pilihan sumber data (Upload File atau Input Langsung)
sumber_data = st.radio(
    "Pilih sumber data:",
    ["Unggah File (CSV/Excel)", "Input Data Langsung / Contoh Bawaan"],
    horizontal=True
)

df = None
sumber_id = ""

if sumber_data == "Unggah File (CSV/Excel)":
    file = st.file_uploader("Unggah file data (CSV/Excel)", type=["csv", "xlsx", "xls"])
    if file is not None:
        try:
            df = baca_file(file)
            sumber_id = file.name
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
else:
    st.info("Ubah atau tambahkan baris data langsung pada tabel di bawah ini:")
    data_default = pd.DataFrame({
        "sampel": [1, 2, 3, 4, 5],
        "luas": [60, 80, 45, 100, 70],
        "kamar": [2, 3, 1, 4, 3],
        "usia": [5, 10, 2, 20, 8],
        "harga": [360, 480, 270, 600, 420]
    })
    df = st.data_editor(data_default, num_rows="dynamic", use_container_width=True)
    sumber_id = "manual_editor"

if df is not None and not df.empty:
    st.subheader("1. Data yang digunakan")
    st.dataframe(df, use_container_width=True)

    # ---- Pilih kolom ----
    kolom_numerik = df.select_dtypes(include="number").columns.tolist()
    if len(kolom_numerik) < 2:
        st.error("Data harus memiliki minimal 2 kolom numerik (fitur dan target).")
        st.stop()

    st.subheader("2. Pilih variabel")
    col1, col2 = st.columns(2)
    with col1:
        kolom_target = st.selectbox(
            "Kolom target (y)", kolom_numerik, index=len(kolom_numerik) - 1
        )
    with col2:
        abaikan = ("sampel", "no", "id", "nomor")
        default_fitur = [
            k for k in kolom_numerik
            if k != kolom_target and k.lower() not in abaikan
        ]
        kolom_fitur = st.multiselect(
            "Kolom fitur (x)",
            [k for k in kolom_numerik if k != kolom_target],
            default=default_fitur,
        )

    pakai_intersep = st.checkbox("Tambahkan intersep (w0)", value=True)

    # ----- Tombol hitung -----
    if st.button("Selesaikan", type="primary"):
        try:
            if not kolom_fitur:
                raise ValueError("Pilih minimal satu kolom fitur.")
            X, y, nama_w = bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep)
            XtX, Xty, w = selesaikan_normal_equation(X, y)
            
            st.session_state["hasil"] = dict(
                sumber=sumber_id,
                fitur=kolom_fitur,
                target=kolom_target,
                X=X,
                y=y,
                nama_w=nama_w,
                XtX=XtX,
                Xty=Xty,
                w=w,
                intersep=pakai_intersep,
            )
        except Exception as e:
            st.session_state.pop("hasil", None)
            st.error(f"Terjadi kesalahan: {e}")

    # ----- Tampilkan hasil -----
    h = st.session_state.get("hasil")
    if h and h["sumber"] == sumber_id:
        X, y, w, nama_w = h["X"], h["y"], h["w"], h["nama_w"]
        w = w + 0.0  # menghindari tampilan -0.0

        st.subheader("3. Langkah perhitungan")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriks desain X**")
            st.dataframe(pd.DataFrame(X, columns=nama_w if h["intersep"] else h["fitur"]))
        with c2:
            st.markdown("**Vektor y**")
            st.dataframe(pd.DataFrame(y, columns=[h["target"]]))

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Matriks $X^T X$**")
            st.dataframe(pd.DataFrame(h["XtX"], index=nama_w, columns=nama_w))
        with c4:
            st.markdown("**Vektor $X^T y$**")
            st.dataframe(pd.DataFrame(h["Xty"], index=nama_w, columns=["X^T y"]))

        st.subheader("4. Solusi w & Persamaan Model")
        hasil_w = pd.DataFrame({"parameter": nama_w, "nilai": np.round(w, 4)})
        st.dataframe(hasil_w, hide_index=True)

        # Persamaan model (diperbaiki agar tanda +/- rapi)
        suku = []
        fitur_list = ([None] + h["fitur"]) if h["intersep"] else h["fitur"]
        for i, (nama, nilai, fitur) in enumerate(zip(nama_w, w, fitur_list)):
            tanda = "" if i == 0 else (" + " if nilai >= 0 else " - ")
            abs_nilai = abs(nilai) if i > 0 else nilai
            if fitur is None:
                suku.append(f"{tanda}{abs_nilai:.4f}")
            else:
                suku.append(f"{tanda}{abs_nilai:.4f}·({fitur})")
        st.info(f"**Model:** {h['target']} = " + "".join(suku))

        # Evaluasi & Residual (Modifikasi sesuai Petunjuk Tugas No. 5)
        st.subheader("5. Evaluasi & Analisis Residual (Galat)")
        y_hat = X @ w
        galat = y - y_hat
        jumlah_galat = float(np.sum(galat))
        sse = float(np.sum(galat ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst > 0 else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{np.sqrt(np.mean(galat ** 2)):.4f}")
        m2.metric("R² (Koefisien Determinasi)", f"{r2:.4f}")
        m3.metric("Jumlah Galat (Σe)", f"{jumlah_galat:.6e}")

        st.markdown("**Perbandingan nilai aktual dan prediksi**")
        df_pred = pd.DataFrame({
            "aktual (y)": y,
            "prediksi (ŷ)": np.round(y_hat, 4),
            "galat (e = y - ŷ)": np.round(galat, 6)
        })
        st.dataframe(df_pred, use_container_width=True)

        st.download_button(
            "Unduh hasil prediksi (CSV)",
            df_pred.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi.csv",
            mime="text/csv",
        )

        # Modifikasi: Grafik Curve Fitting (khusus jika 1 fitur dipilih)
        if len(h["fitur"]) == 1:
            st.subheader("6. Visualisasi Curve Fitting & Residual")
            nama_x = h["fitur"][0]
            x_vals = df[nama_x].to_numpy(dtype=float)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(x_vals, y, color="#1f77b4", label="Data Aktual", zorder=3)
            
            # Garis regresi
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            X_line = np.column_stack([np.ones_like(x_line), x_line]) if h["intersep"] else x_line.reshape(-1, 1)
            y_line = X_line @ w
            ax.plot(x_line, y_line, color="#d62728", linewidth=2, label="Garis Regresi (Curve Fit)")
            
            # Garis residual
            for xi, yi, yhi in zip(x_vals, y, y_hat):
                ax.plot([xi, xi], [yi, yhi], color="gray", linestyle="--", linewidth=1)
                
            ax.set_xlabel(nama_x)
            ax.set_ylabel(h["target"])
            ax.set_title(f"Regresi Linier: {h['target']} vs {nama_x}")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)

        # Prediksi data baru
        st.subheader("7. Prediksi data baru")
        nilai_baru = []
        kolom_in = st.columns(len(h["fitur"]))
        for kol, nama in zip(kolom_in, h["fitur"]):
            nilai_baru.append(kol.number_input(nama, value=float(df[nama].mean())))
            
        x_baru = np.array(([1.0] if h["intersep"] else []) + nilai_baru)
        st.success(f"**Prediksi {h['target']}** = {float(x_baru @ w):.4f}")
else:
    st.info("Silakan unggah file CSV/Excel atau gunakan editor tabel di atas terlebih dahulu.")