import streamlit as st
import numpy as np
import pandas as pd
import time

# Konfigurasi Halaman Web
st.set_page_config(page_title="Visualisasi BiLSTM - Marchel", layout="wide")

st.title("🧠 Simulator Arsitektur BiLSTM")
st.subheader("Klasifikasi Gender Berdasarkan Nama (Studi Kasus: Indonesia)")
st.markdown("---")

# Input Nama dari Pengguna
nama_input = st.text_input("Masukkan Nama untuk Dianalisis:", value="DWI").lower()
tombol_analisis = st.button("🚀 Simulasikan Perjalanan Data")

# Fungsi untuk mewarnai nilai 0 (Dropout) di DataFrame
def highlight_dropout(val):
    color = 'rgba(255, 99, 71, 0.3)' if val == 0 else '' # Warna merah transparan untuk neuron yang mati
    return f'background-color: {color}'

if tombol_analisis and nama_input:
    st.markdown(f"### Menganalisis nama: **{nama_input}**")
    
    col1, col2 = st.columns([1, 2])
    MAX_LEN = 25
    
    # ================= LAYER 1 =================
    with col1:
        st.info("📍 **Layer 1: Input & Padding**\n\nMemecah string menjadi karakter, mengubahnya ke indeks leksikon, dan memastikan panjang array tepat 25.")
    with col2:
        char_dict = {'a': 1, 'i': 2, ',': 3, 'n': 4, 'r': 5, 'u': 6, 's': 7,
                    't': 8, 'm': 9, 'd': 10, 'l': 11, 'h': 12, 'e': 13, 'y': 14,
                    'o': 15, 'f': 16, 'k': 17, 'g': 18, 'p': 19, 'w': 20, 'b': 21,
                    'z': 22, 'v': 23, 'j': 24, 'c': 25, 'q': 26, 'x': 27, '|PAD|': 0}

        chars = list(nama_input.lower())
        indeks = [char_dict.get(c, 0) for c in chars]

        if len(indeks) > MAX_LEN:
            indeks_pad = indeks[:MAX_LEN]
            chars_pad = chars[:MAX_LEN]
        else:
            selisih = MAX_LEN - len(indeks)
            indeks_pad = indeks + [0] * selisih
            chars_pad = chars + ['PAD'] * selisih

        st.write(f"**Panjang Nama Asli:** {len(chars)} karakter")
        df_pad = pd.DataFrame({"Karakter": chars_pad, "Indeks Numerik": indeks_pad})
        st.dataframe(df_pad, height=200, use_container_width=True)
        time.sleep(1)
        
    st.markdown("---")
    
    # ================= LAYER 2 =================
    with col1:
        st.info("📍 **Layer 2: Character Embedding**\n\nMengubah setiap karakter menjadi representasi vektor padat berdimensi 128.")
    with col2:
        dummy_embedding = np.random.randn(MAX_LEN, 128)
        st.write(f"Bentuk Matriks: `{MAX_LEN} karakter × 128 dimensi`")
        st.dataframe(pd.DataFrame(dummy_embedding).iloc[:, :].style.background_gradient(cmap='Blues'), height=200)
        st.caption("*Menampilkan  25 x 128 dimensi.")
        time.sleep(1)
        
    st.markdown("---")
    
    # ================= LAYER 3 =================
    with col1:
        st.warning("📍 **Layer 3: Bidirectional LSTM**\n\nMemproses matriks dari dua arah secara paralel (maju dan mundur). Menghasilkan representasi hidden state 256 dimensi.")
    with col2:
        st.write("🔄 **Proses Forward & Backward berjalan...**")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.success(f"Konteks Dua Arah Terekstrak! (Dimensi output: {MAX_LEN} x 256)")
        time.sleep(1)
        
    st.markdown("---")

    # ================= LAYER 4 (BARU: DROPOUT 1) =================
    with col1:
        st.error("📍 **Layer 4: Dropout (30%)**\n\nLapisan regularisasi pertama. Menonaktifkan 30% koneksi neuron secara acak untuk mencegah model menghafal data (Overfitting).")
    with col2:
        st.write(f"Bentuk Matriks Hasil BiLSTM layer: `{MAX_LEN} × 256`")
        
        # Membuat matriks dummy sesuai dimensi asli
        dummy_bilstm_out = np.random.rand(MAX_LEN, 256)
        
        # PERBAIKAN DI SINI: Gunakan background_gradient, bukan map
        st.dataframe(pd.DataFrame(dummy_bilstm_out).style.background_gradient(cmap='Blues'), height=200)
        
        # Matikan 30% sel secara acak
        st.write(f"Bentuk Matriks setelah dropout layer 1: `{MAX_LEN} × 256`")
        mask_dropout_1 = np.random.rand(MAX_LEN, 256) > 0.3 
        dummy_dropout_1 = dummy_bilstm_out * mask_dropout_1
        
        st.dataframe(pd.DataFrame(dummy_dropout_1).style.map(highlight_dropout), height=200)
        st.caption(f"*Cuplikan matriks {MAX_LEN}x256. Sel berlatar merah (0.000) adalah neuron yang dimatikan.*")
        time.sleep(1)

    st.markdown("---")
    
    # ================= LAYER 5 =================
    with col1:
        st.info("📍 **Layer 5: Global Max Pooling 1D**\n\nMemindai seluruh 25 karakter dan mencomot nilai fitur yang paling dominan/kuat saja.")
    with col2:
        dummy_pooling = np.random.rand(1, 256)
        st.write("Bentuk Array setelah Reduksi Pooling Layer: `1 × 256`")
        st.bar_chart(dummy_pooling[0][:]) 
        time.sleep(1)
        
    st.markdown("---")
    
    # ================= LAYER 6 =================
    with col1:
        st.info("📍 **Layer 6: Dense Layer (64)**\n\nFully Connected Layer dengan aktivasi ReLU. Mengombinasikan fitur-fitur penting menjadi 64 titik penalaran tinggi.")
    with col2:
        dummy_dense = np.random.rand(1, 64)
        st.write("Bentuk Array: `1 × 64`")
        st.dataframe(pd.DataFrame(dummy_dense).style.highlight_max(axis=1, color='lightgreen'))
        time.sleep(1)
        
    st.markdown("---")

    # ================= LAYER 7 (BARU: DROPOUT 2) =================
    with col1:
        st.error("📍 **Layer 7: Dropout (30%)**\n\nSabuk pengaman kedua. Kembali menonaktifkan 30% koneksi dari 64 neuron Dense sebelum tahap klasifikasi akhir.")
    with col2:
        st.write("Bentuk Array: `1 × 64`")
        # Matikan 30% dari 64 neuron
        mask_dropout_2 = np.random.rand(1, 64) > 0.3 
        dummy_dropout_2 = dummy_dense * mask_dropout_2

        st.dataframe(pd.DataFrame(dummy_dropout_2).style.map(highlight_dropout))
        st.caption("*Sel berlatar merah (0.000) adalah neuron yang ditahan/dimatikan untuk sementara.*")
        time.sleep(1)

    st.markdown("---")
    
    # ================= LAYER 8 =================
    with col1:
        st.success("📍 **Layer 8: Output Layer (Sigmoid)**\n\nMemampatkan ke-64 sisa fitur menjadi 1 neuron tunggal. Menghasilkan probabilitas 0 hingga 1.")
    with col2:
        st.write("Menjatuhkan Vonis...")
        time.sleep(1)
        probabilitas = np.random.uniform(0.1, 0.99)
        prediksi = "Perempuan 👩" if probabilitas > 0.5 else "Laki-laki 👨"
        
        st.metric(label="Skor Probabilitas (Sigmoid)", value=f"{probabilitas:.4f}")
        if probabilitas > 0.5:
            st.success(f"### Kesimpulan Klasifikasi: {prediksi}")
        else:
            st.info(f"### Kesimpulan Klasifikasi: {prediksi}")
            
    st.balloons()
