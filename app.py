import streamlit as st
import numpy as np
import pandas as pd
import time

# Konfigurasi Halaman Web
st.set_page_config(page_title="Visualisasi BiLSTM - Marchel", layout="wide")

st.title("🧠 Simulator Arsitektur Hybrid MLE-BiLSTM")
st.subheader("Klasifikasi Gender Berdasarkan Nama (Studi Kasus: Indonesia)")
st.markdown("---")

# Input Nama dari Pengguna
nama_input = st.text_input("Masukkan Nama untuk Dianalisis:", value="DWI").upper()
tombol_analisis = st.button("🚀 Simulasikan Perjalanan Data")

if tombol_analisis and nama_input:
    st.markdown(f"### Menganalisis nama: **{nama_input}**")
    
    # Membuat 2 Kolom untuk Layout (Kiri: Arsitektur, Kanan: Hasil/Matriks)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("📍 **Layer 1: Input & Padding**\n\nMemecah string menjadi karakter dan mengubahnya menjadi indeks angka (Tokenisasi).")
    with col2:
        chars = list(nama_input)
        # Simulasi indeks abjad sederhana (A=1, B=2, dst)
        indeks = [ord(c) - 64 if c.isalpha() else 0 for c in chars]
        st.write("Karakter:", chars)
        st.write("Indeks Numerik:", indeks)
        st.progress(100)
        time.sleep(1) # Efek delay agar interaktif
        
    st.markdown("---")
    
    with col1:
        st.info("📍 **Layer 2: Character Embedding (128 Dim)**\n\nMengubah setiap indeks angka menjadi vektor padat (128 nilai desimal) agar model memahami kedekatan karakter.")
    with col2:
        # Membuat matriks dummy berukuran (Panjang Nama x 128)
        dummy_embedding = np.random.randn(len(chars), 128)
        st.write(f"Bentuk Matriks: `{len(chars)} karakter × 128 dimensi`")
        # Menampilkan sebagian kecil matriks agar tidak penuh layar
        st.dataframe(pd.DataFrame(dummy_embedding).head(len(chars)).iloc[:, :10].style.background_gradient(cmap='Blues'))
        st.caption("*Menampilkan 10 kolom pertama dari 128 dimensi")
        time.sleep(1.5)
        
    st.markdown("---")
    
    with col1:
        st.warning("📍 **Layer 3: Bidirectional LSTM**\n\nMembaca matriks dari dua arah: Maju (Forward) dan Mundur (Backward) secara bersamaan.")
    with col2:
        st.write("🔄 **Proses Forward & Backward berjalan...**")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        # Output BiLSTM biasanya dikali 2 (128 maju + 128 mundur = 256)
        st.success(f"Representasi Konteks Dua Arah Berhasil Diekstrak! (Dimensi: {len(chars)} x 256)")
        time.sleep(1)
        
    st.markdown("---")
    
    with col1:
        st.info("📍 **Layer 4 & 5: Global Max Pooling 1D**\n\nMenyaring hasil BiLSTM dan hanya mengambil nilai sinyal/fitur yang paling kuat (maksimum) dari seluruh karakter.")
    with col2:
        # Simulasi reduksi dari matriks 2D menjadi 1D array
        dummy_pooling = np.random.rand(1, 256)
        st.write("Bentuk Array setelah Pooling: `1 × 256`")
        st.bar_chart(dummy_pooling[0][:50]) # Tampilkan grafik balok untuk 50 fitur pertama
        st.caption("*Grafik 50 sinyal fitur terkuat yang berhasil diekstrak")
        time.sleep(1)
        
    st.markdown("---")
    
    with col1:
        st.info("📍 **Layer 6 & 7: Dense (64) + ReLU & Dropout**\n\nPenalaran tingkat tinggi. ReLU mengubah angka negatif menjadi nol. Dropout mematikan koneksi secara acak untuk mencegah overfitting.")
    with col2:
        dummy_dense = np.random.rand(1, 64)
        st.write("Bentuk Array setelah Penalaran: `1 × 64`")
        st.dataframe(pd.DataFrame(dummy_dense).style.highlight_max(axis=1, color='lightgreen'))
        time.sleep(1)
        
    st.markdown("---")
    
    with col1:
        st.error("📍 **Layer 8: Output Sigmoid**\n\nMemampatkan semua sisa fitur menjadi satu angka probabilitas (0 hingga 1).")
    with col2:
        st.write("Menjatuhkan Vonis...")
        time.sleep(1)
        # Simulasi hasil prediksi
        probabilitas = np.random.uniform(0.1, 0.99)
        prediksi = "Perempuan 👩" if probabilitas > 0.5 else "Laki-laki 👨"
        
        # Tampilan Hasil Akhir yang mencolok
        st.metric(label="Skor Probabilitas (Sigmoid)", value=f"{probabilitas:.4f}")
        if probabilitas > 0.5:
            st.success(f"### Kesimpulan Klasifikasi: {prediksi}")
        else:
            st.info(f"### Kesimpulan Klasifikasi: {prediksi}")
            
    st.balloons() # Efek animasi balon saat selesai
