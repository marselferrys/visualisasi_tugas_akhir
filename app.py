import streamlit as st

st.set_page_config(page_title="Beranda - Hybrid Model", page_icon="🏠", layout="wide")

# Setup memori untuk menyimpan nama agar tersinkronisasi antar halaman
if 'nama_input' not in st.session_state:
    st.session_state.nama_input = "DWI"

st.title("🏠 Simulator Arsitektur Hybrid MLE-BiLSTM")
st.markdown("Selamat datang di purwarupa klasifikasi gender berdasarkan ejaan nama orang Indonesia.")
st.markdown("---")

st.write("Silakan pilih menu di **Sidebar sebelah kiri** untuk memulai simulasi, atau gunakan tombol cepat di bawah ini:")

col1, col2 = st.columns(2)
with col1:
    st.info("**Modul 1: Maximum Likelihood Estimation (MLE)**\n\nFilter tahap pertama menggunakan probabilitas kamus nama berdasarkan kata (Fast-Path).")
    if st.button("📊 Masuk ke Halaman MLE", use_container_width=True):
        st.switch_page("pages/MLE.py") # Melompat ke file MLE

with col2:
    st.warning("**Modul 2: Bidirectional LSTM (BiLSTM)**\n\nTahap mendalam yang membedah nama karakter-demi-karakter untuk mencari pola morfologi tersembunyi.")
    if st.button("🤖 Masuk ke Halaman BiLSTM", use_container_width=True):
        st.switch_page("pages/BiLSTM.py") # Melompat ke file BiLSTM
