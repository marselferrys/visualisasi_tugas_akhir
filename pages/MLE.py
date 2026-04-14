import streamlit as st
import time

st.set_page_config(page_title="Simulasi MLE", page_icon="📊", layout="wide")
st.title("📊 Simulasi Maximum Likelihood Estimation (MLE)")
st.markdown("---")

# Mengambil memori nama dari session_state
nama = st.text_input("Masukkan Nama Lengkap:", value=st.session_state.nama_input).upper()
st.session_state.nama_input = nama # Update memori jika ada ketikan baru

if st.button("🧮 Simulasikan Algoritma MLE"):
    tokens = nama.split()
    st.write(f"**1. Tokenisasi Kata:** `{tokens}`")
    time.sleep(1)
    
    # Mockup Database (Ganti sesuai skripsi Anda)
    tabel_frekuensi = {"BUDI": 0.02, "SANTOSO": 0.05, "SITI": 0.98, "DWI": 0.55}
    skor_tokens = []
    
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
