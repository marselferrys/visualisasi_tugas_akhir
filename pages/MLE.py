import streamlit as st
import time
import pandas as pd

st.set_page_config(page_title="Simulasi MLE", page_icon="📊", layout="wide")

if 'nama_input' not in st.session_state:
    st.session_state.nama_input = "dwi putri"
    
st.title("📊 Simulasi Maximum Likelihood Estimation (MLE)")
st.markdown("---")

# Mengambil memori nama dari session_state
nama = st.text_input("Masukkan Nama Lengkap:", value=st.session_state.nama_input).lower()
st.session_state.nama_input = nama # Update memori jika ada ketikan baru

# Fungsi untuk membaca dan menghitung skor dari Excel
@st.cache_data
def load_mle_database():
    try:
        # Membaca file excel
        df = pd.read_excel("freq_mle_table.xlsx")
        
        # 1. Bersihkan kolom token (Jadikan huruf besar semua dan hilangkan spasi sisa)
        df['token'] = df['token'].astype(str).str.upper().str.strip()
        
        # 2. Hitung Skor Probabilitas (Peluang nama tersebut adalah Perempuan)
        # Rumus: Freq_P dibagi total. Jika total 0 (kosong), beri nilai netral 0.5
        df['skor'] = (df['Freq_P'] / df['total']).fillna(0.5)
        
        # 3. Ubah menjadi Dictionary { 'BUDI': 0.02, 'SITI': 0.98 } untuk pencarian cepat O(1)
        kamus_mle = pd.Series(df['skor'].values, index=df['token']).to_dict()
        
        return kamus_mle
        
    except FileNotFoundError:
        st.error("File 'freq_mle_table.xlsx' tidak ditemukan! Pastikan file sudah ada di folder yang sama.")
        return {}
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses Excel: {e}")
        return {}

# ==========================================
# TAMPILAN HALAMAN MLE
# ==========================================
st.set_page_config(page_title="Simulasi MLE", page_icon="📊", layout="wide")
st.title("📊 Simulasi Maximum Likelihood Estimation (MLE)")
st.markdown("---")

# Memuat database dari excel secara efisien (hanya dibaca 1 kali)
tabel_frekuensi = load_mle_database()

nama = st.text_input("Masukkan Nama Lengkap:", value=st.session_state.nama_input).upper()
st.session_state.nama_input = nama

if st.button("🧮 Simulasikan Algoritma MLE"):
    # Cek apakah tabel berhasil dimuat
    if not tabel_frekuensi:
        st.warning("Menunggu database MLE. Pastikan format Excel Anda sudah benar.")
    else:
        tokens = nama.split()
        st.write(f"**1. Tokenisasi Kata:** `{tokens}`")
        time.sleep(1)
        
        skor_tokens = []
        
        st.write("**2. Pengecekan OOV & Kalkulasi Skor:**")
        for kata in tokens:
            # Mencari kata di dalam dictionary hasil ekstraksi Excel
            if kata in tabel_frekuensi:
                skor = tabel_frekuensi[kata]
                st.success(f"✔️ `{kata}` ditemukan. Skor Peluang Perempuan = {skor:.4f}")
            else:
                skor = 0.5
                st.warning(f"⚠️ `{kata}` adalah OOV (Tidak ada di database). Diberi skor netral = {skor}")
            
            skor_tokens.append(skor)
            time.sleep(1)
            
        rata_rata = sum(skor_tokens) / len(skor_tokens)
        st.markdown("---")
        st.info(f"**3. Rata-rata Skor (MLE Final):** `({str(' + ').join([f'{s:.4f}' for s in skor_tokens])}) / {len(skor_tokens)}` = **{rata_rata:.2f}**")
        
        # Logika rujukan Hybrid 
        if 0.1 <= rata_rata <= 0.9:
            st.error("🚨 **Skor Ambigu!** Nama ini ke model BiLSTM.")
            if st.button("👉 Rujuk nama ini ke Halaman BiLSTM"):
                st.switch_page("pages/2_BiLSTM.py")
        elif rata_rata > 0.9:
            st.success("👩 **Perempuan.**")
        else:
            st.info("👨 **Laki-laki.**")
