import streamlit as st
import time
import pandas as pd

st.set_page_config(page_title="Simulasi MLE", page_icon="📊", layout="wide")

if 'nama_input' not in st.session_state:
    st.session_state.nama_input = "dwi putri"
    
# ==========================================
# FUNGSI LOAD DATABASE
# ==========================================
@st.cache_data
def load_mle_database():
    try:
        df = pd.read_excel("freq_mle_table.xlsx")
        
        # 1. Bersihkan kolom token
        df['token'] = df['token'].astype(str).str.lower().str.strip()
        
        # 2. Hitung Skor Probabilitas
        df['skor'] = (df['Freq_P'] / df['total']).fillna(0.5)
        
        # 3. Jadikan 'token' sebagai index agar pencarian sangat cepat
        df_indexed = df.set_index('token')
        
        # Mengembalikan seluruh dataframe, bukan cuma dictionary
        return df_indexed
        
    except FileNotFoundError:
        st.error("File 'freq_mle_table.xlsx' tidak ditemukan!")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses Excel: {e}")
        return None

# ==========================================
# TAMPILAN HALAMAN MLE
# ==========================================
st.title("📊 Simulator Maximum Likelihood Estimation (MLE)")
st.markdown("---")

# Memuat tabel frekuensi token nama dari excel
df_frekuensi = load_mle_database()

nama = st.text_input("Masukkan Nama Lengkap:", value=st.session_state.nama_input).lower()
st.session_state.nama_input = nama

if st.button("🧮 Simulasikan Algoritma MLE"):
    if df_frekuensi is None:
        st.warning("Database tidak tersedia.")
    else:
        tokens = nama.split()
        st.write(f"### 1. Tokenisasi Kata")
        st.info(f"Input dipecah menjadi **{len(tokens)}** token: `{tokens}`")
        time.sleep(1)
        
        st.markdown("---")
        st.write("### 2. Ekstraksi Data dari Tabel Frekuensi (Korpus)")
        st.write("Sistem mencari token-token tersebut di dalam *tabel* frekuensi token nama.")
        
        # Mencari kata yang ada di database untuk ditampilkan di tabel
        tokens_ditemukan = [t for t in tokens if t in df_frekuensi.index]
        
        if tokens_ditemukan:
            # Mengambil baris data khusus untuk token yang diketik user
            df_tampil = df_frekuensi.loc[tokens_ditemukan, ['Freq_L', 'Freq_P', 'total', 'skor']]
            
            # Menampilkan tabel dengan highlight pada kolom skor
            st.dataframe(
                df_tampil.style.highlight_max(subset=['skor'], color='rgba(144, 238, 144, 0.5)')
                               .format({"skor": "{:.4f}"}), 
                use_container_width=True
            )
        else:
            st.warning("Tidak ada satu pun token yang terdaftar di dalam Tabel Frekuensi (Semuanya OOV).")
            
        time.sleep(1)
        
        st.markdown("---")
        st.write("### 3. Detail Kalkulasi Skor Peluang (MLE) per Token")
        st.write("Menggunakan rumus MLE:")
        st.latex(r"\text{MLE(female)} = \frac{\text{count(female)}}{\text{count(female)} + \text{count(male)}}")
        
        skor_tokens = []
        
        # Membuat kolom untuk menampilkan kalkulasi agar rapi
        kolom_kalkulasi = st.columns(len(tokens))
        
        for i, kata in enumerate(tokens):
            with kolom_kalkulasi[i]:
                st.markdown(f"**Token: `{kata.upper()}`**")
                
                if kata in df_frekuensi.index:
                    row = df_frekuensi.loc[kata]
                    freq_p = int(row['Freq_P'])
                    freq_l = int(row['Freq_L'])
                    total = int(row['total'])
                    skor = row['skor']
                    
                    st.write(f"- Freq Perempuan: `{freq_p}`")
                    st.write(f"- Freq Laki-laki: `{freq_l}`")
                    st.write(f"- Total Muncul: `{total}`")
                    
                    # Menampilkan rumus matematika menggunakan LaTeX
                    st.latex(r"\text{Skor} = \frac{\text{Freq P}}{\text{Freq P} + \text{Freq L}}")
                    st.latex(rf"\frac{{{freq_p}}}{{{total}}} = {skor:.4f}")
                    st.success(f"**Skor = {skor:.4f}**")
                    
                else:
                    skor = 0.5
                    st.write("- Freq Perempuan: `0`")
                    st.write("- Freq Laki-laki: `0`")
                    st.write("- Total Muncul: `0`")
                    
                    st.error("⚠️ *Out of Vocabulary* (OOV)")
                    st.latex(r"\text{Penalti Netral} = 0.5000")
                    st.warning(f"**Skor = 0.5000**")
                    
                skor_tokens.append(skor)
                time.sleep(0.5)
                
        st.markdown("---")
        st.write("### 4. Keputusan Akhir (Average All Scores)")
        rata_rata = sum(skor_tokens) / len(skor_tokens)
        
        # Menampilkan rumus rata-rata
        rumus_rata_rata = " + ".join([f"{s:.4f}" for s in skor_tokens])
        # Tampilkan penjumlahan skor
        st.latex(rf"\sum \text{{Skor MLE}} = {rumus_rata_rata}")

        # Jumlah data
        # st.latex(rf"n = {len(skor_tokens)}")

        # Perhitungan akhir
        st.latex(rf"\text{{Rata-rata}} = \frac{{{rumus_rata_rata}}}{{{len(skor_tokens)}}} = {rata_rata:.2f}")
        st.success(f"**MLE Score untuk {nama}  = {rata_rata:.2f}**")
        st.caption("Threshold: Perempuan > 0.90 | Laki-laki < 0.10 | Ambigu 0.10 - 0.90")
        
        # Logika rujukan Hybrid 
        if 0.1 <= rata_rata <= 0.9:
            st.error(f"🚨 **Skor {rata_rata:.2f} bersifat Ambigu!**")
        elif rata_rata > 0.9:
            st.success("👩 **Perempuan.**")
        else:
            st.info("👨 **Laki-laki.**")
