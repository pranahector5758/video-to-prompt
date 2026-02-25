import streamlit as st
import cv2
from PIL import Image
import tempfile
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# ⚙️ KONFIGURASI HALAMAN & MODEL
# ==========================================
st.set_page_config(page_title="Video to Prompt Generator", page_icon="🎬", layout="wide")

# Menggunakan cache agar model AI tidak di-download/di-load berulang kali setiap klik
@st.cache_resource(show_spinner="Memuat Model AI (Ini mungkin memakan waktu saat pertama kali dijalankan)...")
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# ==========================================
# 🛠️ FUNGSI-FUNGSI UTAMA
# ==========================================
def save_uploaded_file(uploaded_file):
    """Menyimpan file video sementara ke sistem agar bisa dibaca oleh OpenCV."""
    try:
        # Membuat file sementara dengan ekstensi .mp4
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"Gagal menyimpan file video: {e}")
        return None

def extract_frames(video_path, max_frames):
    """Mengekstrak 1 frame setiap detiknya (1 fps) dari video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: OpenCV tidak dapat membaca video ini.")
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback jika fps gagal terdeteksi
    frame_interval = int(fps) # Ambil 1 frame per detik

    success, image = cap.read()
    count = 0
    extracted_count = 0

    while success and extracted_count < max_frames:
        # Hanya ambil frame sesuai interval (1 fps)
        if count % frame_interval == 0:
            # Konversi warna dari BGR (OpenCV) ke RGB (Standar Gambar/PIL)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            frames.append(pil_image)
            extracted_count += 1
            
        success, image = cap.read()
        count += 1

    cap.release()
    return frames

def analyze_frame(image, processor, model):
    """Menganalisis gambar menggunakan model BLIP untuk menghasilkan deskripsi teks."""
    try:
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        description = processor.decode(out[0], skip_special_tokens=True)
        return description
    except Exception as e:
        return f"[Error analisis frame: {e}]"

def generate_prompt(descriptions):
    """Menggabungkan hasil analisis menjadi satu prompt yang rapi."""
    # Menghapus deskripsi yang sama persis secara berurutan agar tidak spam
    unique_desc = []
    for desc in descriptions:
        if not unique_desc or desc != unique_desc[-1]:
            unique_desc.append(desc)
    
    prompt = "Berikut adalah urutan kejadian dalam video berdasarkan analisis visual:\n\n"
    for i, desc in enumerate(unique_desc, 1):
        prompt += f"Adegan {i}: {desc.capitalize()}.\n"
    
    prompt += "\nBerdasarkan urutan adegan di atas, buatkan deskripsi naratif yang utuh, detail, dan sinematik."
    return prompt

# ==========================================
# 🖥️ ANTARMUKA PENGGUNA (UI)
# ==========================================
def main():
    st.title("🎬 Video to Prompt Generator")
    st.markdown("Unggah video, dan AI akan 'menontonnya' lalu membuatkan prompt deskriptif secara otomatis!")

    # Load Model di awal
    processor, model = load_model()

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload Video")
        st.info("💡 **Instruksi:**\n1. Upload video (Maks 200MB).\n2. Atur jumlah frame.\n3. Klik Generate.\n4. Salin/Download hasilnya.")
        
        uploaded_file = st.file_uploader("Pilih file video", type=['mp4', 'mov', 'avi'])
        
        st.divider()
        st.header("⚙️ Pengaturan")
        max_frames = st.slider(
            "Maksimal Frame yang Dianalisis", 
            min_value=3, max_value=20, value=5,
            help="Semakin banyak frame, semakin akurat promptnya, tapi prosesnya akan memakan waktu lebih lama."
        )

    # Main Area
    if uploaded_file is not None:
        # Validasi ukuran (Streamlit secara default membatasi 200MB, ini pengecekan ekstra)
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("⚠️ Ukuran file melebihi 200MB. Silakan kompres video Anda terlebih dahulu.")
            return

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📺 Preview Video")
            st.video(uploaded_file)

        with col2:
            st.subheader("🤖 Hasil Generasi Prompt")
            
            if st.button("✨ Generate Prompt", type="primary", use_container_width=True):
                # 1. Simpan video sementara
                temp_video_path = save_uploaded_file(uploaded_file)
                
                if temp_video_path:
                    try:
                        # 2. Ekstraksi Frame
                        with st.spinner(f"🎞️ Mengekstrak {max_frames} frame dari video..."):
                            frames = extract_frames(temp_video_path, max_frames)
                        
                        if not frames:
                            st.warning("Tidak ada frame yang berhasil diekstrak. Coba video lain.")
                        else:
                            st.success(f"Berhasil mengekstrak {len(frames)} frame!")
                            
                            # Tampilkan thumbnail kecil frame yang diekstrak
                            st.image(frames, width=80, caption=[f"F-{i+1}" for i in range(len(frames))])

                            # 3. Analisis Frame (dengan Progress Bar)
                            descriptions = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, frame in enumerate(frames):
                                status_text.text(f"🔍 Menganalisis frame {i+1} dari {len(frames)}...")
                                desc = analyze_frame(frame, processor, model)
                                descriptions.append(desc)
                                progress_bar.progress((i + 1) / len(frames))
                            
                            status_text.text("✅ Analisis selesai!")

                            # 4. Generate Prompt Akhir
                            final_prompt = generate_prompt(descriptions)
                            
                            # Tampilkan hasil di text area (bisa diedit) dan code block (bisa dicopy)
                            st.divider()
                            st.markdown("### 📝 Prompt Anda Siap!")
                            
                            # Fitur Copy bawaan Streamlit (ikon copy ada di pojok kanan atas kotak ini)
                            st.code(final_prompt, language="markdown")
                            
                            # Opsi Download sebagai .txt
                            st.download_button(
                                label="⬇️ Download Prompt (.txt)",
                                data=final_prompt,
                                file_name="video_prompt.txt",
                                mime="text/plain",
                                use_container_width=True
                            )

                    finally:
                        # Membersihkan file sementara agar hardisk tidak penuh
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)

if __name__ == "__main__":
    main()