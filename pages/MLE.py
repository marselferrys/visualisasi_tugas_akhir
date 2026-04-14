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

# Fungsi untuk membaca database dari Excel
@st.cache_data
def load_mle_database():
    try:
        # Membaca file excel
        df = pd.read_excel("freq_mle_table.xlsx")
        
        # Memastikan nama kolom seragam (huruf besar) dan menghapus spasi
        df['nama'] = df['nama'].astype(str).str.upper().str.strip()
        
        # Mengubah DataFrame menjadi Dictionary (Key: Nama, Value: Skor)
        return pd.Series(df.skor.values, index=df.nama).to_dict()
    except Exception as e:
        st.error(f"Gagal memuat file Excel: {e}")
        return {}

# Memanggil fungsi load
tabel_frekuensi = load_mle_database()

# Logika pengecekan tetap sama
skor_tokens = []
if st.button("🧮 Simulasikan Algoritma MLE"):
    if not tabel_frekuensi:
        st.warning("Database MLE kosong. Pastikan file freq_mle_table.xlsx sudah diunggah.")
    else:
        tokens = nama.split()
    st.write("**2. Pengecekan OOV (Tabel Frekuensi):**")
    for kata in tokens:
        if kata in tabel_frekuensi:
            skor = tabel_frekuensi[kata]
            st.success(f"✔️ `{kata}` ditemukan. Skor = {skor}")
        else:
            skor = 0.5
            st.warning(f"⚠️ `{kata}` adalah OOV. Diberi skor netral = {skor}")
        skor_tokens.append(skor)
        time.sleep(1)
        
    rata_rata = sum(skor_tokens) / len(skor_tokens)
    st.markdown("---")
    st.info(f"**3. Rata-rata Skor (MLE Final):** `({str(' + ').join(map(str, skor_tokens))}) / {len(skor_tokens)}` = **{rata_rata:.4f}**")
    
    if 0.1 <= rata_rata <= 0.9:
        st.error("🚨 **Skor Ambigu!** Nama ini ke model BiLSTM.")
        if st.button("👉 ke Halaman BiLSTM"):
            st.switch_page("pages/2_BiLSTM.py")
    elif rata_rata > 0.9:
        st.success("👩 **Perempuan.** (Selesai)")
    else:
        st.info("👨 **Laki-laki.** (Selesai)")
