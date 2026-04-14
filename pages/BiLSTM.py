import streamlit as st
import numpy as np
import pandas as pd
import time
from gradio_client import Client

# Konfigurasi Halaman Web
st.set_page_config(page_title="Visualisasi BiLSTM - Marchel", layout="wide")

st.title("🧠 Simulator Arsitektur BiLSTM")
st.subheader("Klasifikasi Gender Berdasarkan Nama (Studi Kasus: Nama Orang Indonesia)")
st.markdown("---")

# Input Nama dari Pengguna
nama_input = st.text_input("Masukkan Nama untuk Dianalisis:", value="dwi putri").lower()
tombol_analisis = st.button("🚀 Simulasikan Model BiLSTM")

# Fungsi untuk mewarnai nilai 0 (Dropout) di DataFrame

# def highlight_dropout(val):
#    color = 'rgba(255, 99, 71, 0.3)' if val == 0 else '' # Warna merah transparan untuk neuron yang mati
#    return f'background-color: {color}'

# Fungsi untuk memberi highlight pada angka yang diubah menjadi 0 untuk Dropout
def highlight_dropout(val):
    color = '#ff0000' if val == 0 else ''
    return f'background-color: {color}; color: white;'

# Fungsi untuk memberi highlight pada angka yang diubah menjadi 0 untuk Relu
def highlight_zero_relu(val):
    return 'background-color: rgba(255, 255, 0, 0.3); color: white;' if val == 0 else ''

# Fungsi untuk memberi warna teks merah pada angka negatif
def color_negative_red(val):
    return 'color: #ff4b4b; font-weight: bold;' if val < 0 else ''

if tombol_analisis and nama_input:
    st.markdown(f"### Menganalisis nama: **{nama_input}**")
    
    col1, col2 = st.columns([1, 2])
    MAX_LEN = 25
    
    # ================= LAYER 1 =================
    with col1:
        st.info(f"📍 **Layer 1: Input & Padding**\n\nMemecah string menjadi karakter, mengubahnya ke indeks leksikon asli, dan memastikan panjang array tepat {MAX_LEN}.")
    with col2:
        # Leksikon asli sesuai permintaan Anda
        char_dict = {'a': 1, 'i': 2, ' ': 3, 'n': 4, 'r': 5, 'u': 6, 's': 7,
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
            # Menggunakan '|PAD|' sebagai penanda visual di tabel
            chars_pad = chars + ['|PAD|'] * selisih

        st.write(f"**Panjang Nama Asli:** {len(chars)} karakter")
        df_pad = pd.DataFrame({"Karakter": chars_pad, "Indeks Numerik": indeks_pad})
        st.dataframe(df_pad, height=200, use_container_width=True)
        time.sleep(1)
        
    st.markdown("---")
    
    # ================= LAYER 2 =================
    with col1:
        st.info("📍 **Layer 2: Character Embedding**\n\nMengubah setiap indeks menjadi vektor padat 128-dimensi menggunakan *Lookup Table*.")
    with col2:
        # Menetapkan seed agar nilai acak tetap konsisten untuk kebutuhan visualisasi
        np.random.seed(42)
        
        # Membuat Bobot Embedding Utama (Lookup Table)
        # Ukuran: jumlah karakter unik dalam leksikon (28) x dimensi (128)
        ukuran_kamus = len(char_dict)
        dimensi_embedding = 128
        bobot_asli = np.random.randn(ukuran_kamus, dimensi_embedding)
        
        # Mengambil baris dari bobot_asli berdasarkan indeks_pad
        dummy_embedding = np.array([bobot_asli[idx] for idx in indeks_pad])
        
        st.write(f"Bentuk Matriks dari Character Embedding Layer: `{MAX_LEN} karakter × 128 dimensi`")
        
        # Menampilkan 10 kolom pertama untuk keterbacaan
        df_emb = pd.DataFrame(dummy_embedding)
        st.dataframe(df_emb.iloc[:, :].style.background_gradient(cmap='Blues'), height=200)
        
        st.caption(f"Menampilkan Matriks *{MAX_LEN} x 128*.")
        
        # Reset seed agar tidak mempengaruhi random layer berikutnya
        np.random.seed(None)
        time.sleep(1)
        
    st.markdown("---")
    
    # ================= LAYER 3 =================
    with col1:
        st.warning("📍 **Layer 3: Bidirectional LSTM**\n\nMemproses matriks dari dua arah secara paralel (maju dan mundur). Menghasilkan representasi hidden state 256 dimensi.")
    with col2:
        st.write("BiLSTM Layer")
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
        st.write(f"Bentuk Matriks Hasil BiLSTM layer: `{MAX_LEN} × 256`, Concat 2 LSTM Layer (128+128)")
        
        # Membuat matriks dummy sesuai dimensi asli
        dummy_bilstm_out = np.random.rand(MAX_LEN, 256)
        
        st.dataframe(pd.DataFrame(dummy_bilstm_out).style.background_gradient(cmap='Blues'), height=200)
        
        # Matikan 30% sel secara acak
        st.write(f"Bentuk Matriks setelah dropout layer 1: `{MAX_LEN} × 256`")
        mask_dropout_1 = np.random.rand(MAX_LEN, 256) > 0.3 
        dummy_dropout_1 = dummy_bilstm_out * mask_dropout_1
        
        st.dataframe(pd.DataFrame(dummy_dropout_1).style.map(highlight_dropout), height=200)
        st.caption(f"*Cuplikan matriks {MAX_LEN}x256. Sel berlatar merah (0.000) adalah neuron yang dinonaktifkan.*")
        time.sleep(1)

    st.markdown("---")
    
    # ================= LAYER 5 =================
    with col1:
        # Menggunakan variabel MAX_LEN agar teks dinamis menyesuaikan panjang padding
        st.info(f"📍 **Layer 5: Global Max Pooling 1D**\n\nMemindai seluruh {MAX_LEN} karakter dan mencomot nilai fitur yang paling dominan saja.")
    with col2:
        st.write("Pooling Layer (**GlobalMaxPooling1D**)")
        dummy_pooling = np.max(dummy_dropout_1, axis=0, keepdims=True)
        st.write("1️⃣ **Bentuk Matriks setelah Reduksi Pooling Layer:** `1 × 256`")
        st.bar_chart(dummy_pooling[0][:]) 
        st.caption("*Mengambil nilai fitur paling dominan dari tiap karakter*")
        st.write("2️⃣ **Bentuk Array 1D:** `1 × 256`")
        st.dataframe(pd.DataFrame(dummy_pooling).style.background_gradient(cmap='Blues'))
        time.sleep(1)
        
    st.markdown("---")
    
    # ================= LAYER 6 =================
    with col1:
        st.info("📍 **Layer 6: Dense Layer (64)**\n\nFully Connected Layer dengan aktivasi ReLU. Mengombinasikan fitur-fitur penting menjadi 64 titik penalaran tinggi.")
    with col2:
        # Menghasilkan angka acak antara -1 hingga 1 agar terdapat nilai negatif
        raw_dense = np.random.uniform(-1, 1, (1, 64))
        
        st.write("Dense Layer (64)")
        st.write("1️⃣ **Raw Dense Output:** `1 × 64`")
            
        st.dataframe(pd.DataFrame(raw_dense).style.map(color_negative_red))
        st.caption("*Angka berwarna merah adalah nilai negatif (sinyal lemah/noise).*")

        time.sleep(1)
        
        st.line_chart(raw_dense[0])
        st.caption("*Grafik Raw Dense Output: Lembah yang turun di bawah garis 0 adalah sinyal negatif (noise).*")
        time.sleep(1)
        
        # Menerapkan fungsi matematis ReLU: max(0, x)
        relu_dense = np.maximum(0, raw_dense)
        
        st.write("2️⃣ **Setelah Fungsi Aktivasi ReLU:** `1 × 64`")
        st.dataframe(pd.DataFrame(relu_dense).style.map(highlight_zero_relu))
        st.caption("*Sinyal negatif diubah menjadi 0.0000 (Sel berlatar hijau).*")
        
        st.line_chart(relu_dense[0])
        st.caption("*Grafik setelah ReLU: Lembah negatif terpotong rata menjadi 0.0000. Hanya sinyal positif yang diteruskan.*")
        time.sleep(1)
        
    st.markdown("---")

    # ================= LAYER 7 (DROPOUT 2) =================
    with col1:
        st.error("📍 **Layer 7: Dropout (30%)**\n\nKembali menonaktifkan 30% koneksi dari 64 neuron Dense sebelum tahap klasifikasi akhir.")
    with col2:
        st.write("Bentuk Array setelah Dropout Layer 2: `1 × 64`")
        # Matikan 30% dari 64 neuron
        mask_dropout_2 = np.random.rand(1, 64) > 0.3 
        dummy_dropout_2 = relu_dense * mask_dropout_2

        st.dataframe(pd.DataFrame(dummy_dropout_2).style.map(highlight_dropout))
        st.caption("*Sel berlatar merah (0.000) adalah neuron yang dinonaktifkan.*")
        time.sleep(1)

    st.markdown("---")
    
    # ================= LAYER 8 =================
    with col1:
        st.success("📍 **Layer 8: Output Layer (Sigmoid)**\n\nMemampatkan ke-64 sisa fitur menjadi 1 neuron tunggal. Menghasilkan probabilitas 0 hingga 1.")
    with col2:
        st.write("Hasil Klasifikasi...")
        
        try:
            # Memanggil Client Hugging Face secara langsung
            client = Client("marselferrys/indo_name-gender-prediction")
            
            # Menembak API. Hasil kembalian berupa tuple
            result = client.predict(
                nama_input, 
                api_name="/predict"
            )
            
            # 1. Ekstraksi out_gender (indeks 0) dan bersihkan teksnya
            gender_api_result = str(result[0]).strip().upper()
            
            # 2. Ekstraksi out_conf (indeks 1) dan pastikan formatnya float (angka desimal)
            probabilitas_api = float(result[1])

            # Tampilkan hasil akhir klasifikasi
            if gender_api_result == "F":
                st.success("### Perempuan 👩")
            elif gender_api_result == "M":
                st.info("### Laki-laki 👨")
            else:
                st.warning(f"### Kesimpulan: Tidak Diketahui ({gender_api_result})")
            
            # Tampilkan metrik probabilitas asli dari model
            st.metric(label="Skor Probabilitas", value=f"{probabilitas_api:.2f}")
                
        except Exception as e:
            # Menampilkan pesan error jika terjadi masalah koneksi ke server HF
            st.error(f"Gagal menghubungi model di Hugging Face. Detail: {e}")
            
    st.balloons()
            
    st.balloons()
