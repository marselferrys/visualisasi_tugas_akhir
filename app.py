import streamlit as st

st.set_page_config(page_title="Beranda - Hybrid Model", page_icon="🏠", layout="wide")

# Setup memori untuk menyimpan nama agar tersinkronisasi antar halaman
if 'nama_input' not in st.session_state:
    st.session_state.nama_input = "dwi putri"

st.title("🏠 Simulator Pemrosesan Data Nama Pada Model MLE dan BiLSTM")
st.markdown("""Diharapkan melalui platform ini, proses dibalik model yang kompleks dapat diubah menjadi simulasi *step-by-step* yang intuitif. 
Tujuan utamanya adalah untuk memberikan **transparansi penuh** terhadap alur pemrosesan data nama, 
sehingga setiap keputusan yang diambil oleh model dapat dipahami dan dijelaskan secara visual.""")
st.markdown("---")

st.write("Silakan pilih menu di **Sidebar sebelah kiri** untuk memulai simulasi, atau gunakan tombol cepat di bawah ini:")

col1, col2 = st.columns(2)
with col1:
    st.info("**Modul 1: Maximum Likelihood Estimation (MLE)**\n\nFilter tahap pertama mmenggunakan skor probabilitas yang dihitung dari frekuensi kemunculan token nama dalam tabel frekuensi token nama (korpus) (Fast-Path).")
    if st.button("📊 Masuk ke Halaman MLE", use_container_width=True):
        st.switch_page("pages/MLE.py") # Melompat ke file MLE

with col2:
    st.warning("**Modul 2: Bidirectional LSTM (BiLSTM)**\n\nTahap mendalam yang membedah nama karakter-demi-karakter untuk mencari pola morfologi tersembunyi.")
    if st.button("🤖 Masuk ke Halaman BiLSTM", use_container_width=True):
        st.switch_page("pages/BiLSTM.py") # Melompat ke file BiLSTM
