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
        st.info("📍 **Layer 1: Input & Padding**\n\nMemecah string menjadi karakter, mengubahnya menjadi indeks angka (sesuai leksikon), dan memastikan panjang array tepat 30 (Padding/Truncating).")
    with col2:
        # 1. Definisi Kamus Leksikon (Sesuai dengan gambar dari model BiLSTM Anda)
        char_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 
                     'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 
                     'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 
                     'z': 26, ' ': 27, '-': 28, "'": 29}
        MAX_LEN = 30

        # 2. Tokenisasi (Ubah teks ke lowercase karena kamus menggunakan huruf kecil)
        chars = list(nama_input.lower())
        
        # Ambil nilai dari kamus, jika karakter aneh/tidak ada (OOV), berikan nilai 0 sementara
        indeks = [char_dict.get(c, 0) for c in chars]

        # 3. Logika Truncating (Potong) dan Padding (Tambah 0)
        if len(indeks) > MAX_LEN:
            # Jika nama lebih dari 30 huruf, potong paksa
            indeks_pad = indeks[:MAX_LEN]
            chars_pad = chars[:MAX_LEN]
        else:
            # Jika kurang dari 30 huruf, tambahkan angka 0 dan label 'PAD' di akhir
            selisih = MAX_LEN - len(indeks)
            indeks_pad = indeks + [0] * selisih
            chars_pad = chars + ['PAD'] * selisih

        st.write(f"**Panjang Nama Asli:** {len(chars)} karakter")
        st.write("**Karakter (Padded):**", chars_pad)
        st.write(f"**Indeks Numerik (Padded {MAX_LEN}):**", indeks_pad)
        
        st.progress(100)
        time.sleep(1) # Efek delay agar interaktif
        
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
