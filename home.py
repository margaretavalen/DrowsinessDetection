import streamlit as st

# Penjelasan tentang Sistem Deteksi Kantuk
def explain_drowsiness_detection():
    # Deskripsi aplikasi
    st.markdown(
        """
        **Selamat datang** di aplikasi kami yang didedikasikan untuk mendeteksi kantuk ketika berkendara.  
        Aplikasi ini didedikasikan untuk meningkatkan keselamatan dengan memantau dan mendeteksi tanda-tanda kantuk secara real-time.  

        Kami menggunakan algoritma CNN (Convolutional Neural Network) untuk mendeteksi wajah secara efektif dan akurat. 
        CNN adalah algoritma berbasis pembelajaran mendalam yang dirancang khusus untuk pengolahan citra dan pengenalan pola.  

        Dengan deteksi ini, kami berharap dapat: 
        - Mengurangi risiko kecelakaan dengan mendeteksi tanda-tanda kantuk atau ketidaksadaran.  
        - Meningkatkan keselamatan pengguna selama berkendara.  
        """
    )

    # Gambar dengan caption
    st.image('images/drowsiness.jpg', caption='Drowsiness Detection System', use_container_width=True)

# Penjelasan tentang EAR (Eye Aspect Ratio)
def explain_ear_tab():
    st.markdown(
        """
        ## Apa itu EAR (Eye Aspect Ratio)?
        
        **EAR (Eye Aspect Ratio)** adalah rasio yang digunakan untuk mengukur seberapa terbuka atau tertutupnya mata seseorang. 
        Rasio ini dihitung dengan mengukur jarak antara titik-titik landmark yang ada di sekitar mata pada gambar atau video wajah. 
        EAR digunakan untuk mendeteksi apakah seseorang mengantuk berdasarkan apakah mata mereka terbuka atau tertutup.
        
        ### Cara Menghitung EAR:
        EAR dihitung dengan rumus:
        
        \[
        EAR = \frac{ \text{jarak vertikal antara titik atas dan bawah} }{ \text{jarak horizontal antara titik kiri dan kanan} }
        \]
        
        - **Mata terbuka**: Jika EAR lebih besar dari nilai ambang batas (0.3), mata dianggap terbuka.
        - **Mata tertutup**: Jika EAR lebih rendah dari nilai ambang batas, mata dianggap tertutup, yang menunjukkan potensi kantuk atau ketidaksadaran.
        
        ### Fungsi EAR dalam Deteksi Kantuk:
        - **Mengukur mata tertutup**: Jika EAR lebih rendah dari ambang batas yang ditentukan, ini menunjukkan bahwa mata tertutup.
        - **Alarm Deteksi Kantuk**: Jika EAR tetap rendah untuk beberapa waktu (3 detik), sistem akan mendeteksi kantuk dan mengaktifkan alarm untuk memperingatkan pengguna.
        
        ### Mengapa EAR Penting?
        - Dengan memantau EAR, aplikasi dapat mendeteksi tanda-tanda kantuk atau kelelahan, yang penting untuk keselamatan, terutama saat berkendara atau melakukan aktivitas yang memerlukan kewaspadaan.
        """
    )

def explain_vggface_tab():
    st.markdown(
        """
        ## Apa itu VGGFace?

        **VGGFace** adalah model Convolutional Neural Network (CNN) yang dikembangkan oleh Visual Geometry Group (VGG) di University of Oxford, khusus untuk pengenalan wajah. VGGFace dirancang untuk mengenali wajah manusia dalam gambar atau video dengan tingkat akurasi yang sangat tinggi.

        ### Bagaimana VGGFace Bekerja dalam Aplikasi Deteksi Kantuk?
        Dalam aplikasi deteksi kantuk, **VGGFace** digunakan untuk mendeteksi wajah pengguna secara akurat. Setelah wajah terdeteksi, sistem akan memantau mata untuk melihat apakah mereka tertutup atau tetap terbuka. Deteksi mata ini menggunakan rasio EAR (Eye Aspect Ratio).

        ### Arsitektur VGGFace:
        - VGGFace menggunakan **VGGNet**, yang terkenal dengan kedalamannya yang besar (banyak lapisan konvolusi). Model ini telah dilatih dengan dataset wajah yang sangat besar, termasuk variasi pencahayaan, ekspresi, dan sudut wajah.
        - VGGFace sangat efektif untuk mendeteksi wajah manusia meskipun ada variasi pada pose atau kondisi pencahayaan.

        ### Keunggulan Menggunakan VGGFace dalam Deteksi Kantuk:
        - **Akurasi Tinggi:** VGGFace mampu mengenali wajah meskipun ada banyak perbedaan pada pose atau pencahayaan.
        - **Deteksi Wajah Real-time:** VGGFace bekerja dengan sangat baik dalam mendeteksi wajah dalam waktu nyata, yang penting dalam aplikasi deteksi kantuk.
        - **Ekstraksi Fitur Wajah:** Setelah wajah terdeteksi, VGGFace dapat membantu mengekstrak fitur wajah untuk kemudian digunakan dalam pemantauan mata (untuk penghitungan EAR).

        ### Mengapa VGGFace Penting dalam Aplikasi Ini?
        Dengan menggunakan VGGFace untuk mendeteksi wajah, aplikasi dapat dengan cepat dan akurat memantau mata pengguna. Jika mata tertutup untuk waktu tertentu, aplikasi akan memberi peringatan atau alarm untuk mencegah kecelakaan akibat kantuk.
        """
    )

# Penjelasan fitur deteksi kantuk
def explain_drowsiness_detection_features():
    st.markdown(
        """
        ### Fitur Deteksi Kantuk
        Aplikasi ini menggunakan algoritma CNN dengan model **VGGFace** dan **EAR** untuk mendeteksi apakah mata pengguna tertutup atau tidak. 
        Jika mata tertutup lebih lama dari batas waktu yang ditentukan, alarm akan berbunyi untuk memperingatkan pengguna.

        #### Fitur Webcam
        Aplikasi ini dapat menggunakan webcam perangkat Anda untuk memantau wajah dan mata secara real-time. Webcam akan menangkap gambar wajah pengguna, dan sistem akan memproses gambar tersebut untuk mengukur **Eye Aspect Ratio (EAR)**. 
        Jika mata pengguna tertutup lebih lama dari waktu yang telah ditentukan (3 detik), aplikasi akan memberi peringatan.

        - **Kelebihan**: Memungkinkan deteksi kantuk secara langsung tanpa memerlukan gambar yang sudah ada.
        - **Cara Menggunakan**: Tekan tombol **Start Detection** untuk memulai pemantauan wajah melalui webcam.

        #### Fitur Upload Gambar
        Selain menggunakan webcam, aplikasi ini juga memungkinkan pengguna untuk mengunggah gambar untuk deteksi kantuk. Anda dapat mengunggah gambar wajah Anda, dan sistem akan memproses gambar tersebut untuk mengukur **EAR** dan mendeteksi tanda-tanda kantuk.

        - **Kelebihan**: Berguna ketika webcam tidak tersedia atau pengguna ingin menguji deteksi dengan gambar yang sudah ada.
        - **Cara Menggunakan**: Pilih gambar dari perangkat Anda menggunakan tombol **Upload Image**, dan aplikasi akan melakukan pemantauan berdasarkan gambar yang diunggah.

        Dengan kedua fitur ini, aplikasi memberikan fleksibilitas untuk mendeteksi kantuk baik melalui pemantauan langsung dengan webcam maupun gambar yang diunggah sebelumnya.
        """
    )


# Fungsi untuk menampilkan halaman utama dengan Tab
def home_page():
    # Menampilkan judul halaman
    st.markdown(
        """
        <h1 style="text-align: center; color: #ffffff; font-size: 40px; font-weight: bold;">
        Drowsiness Detection System
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Menu pilihan untuk penjelasan
    menu_selection = st.selectbox(
        "Pilih Penjelasan:",
        ["Sistem Deteksi Kantuk", "Fitur Deteksi Kantuk", "Apa itu VGGFace?", "Apa itu EAR?", "Cara Menggunakan"],
        index=0  # Menetapkan indeks 0 agar "Penjelasan Sistem Deteksi Kantuk" ditampilkan pertama kali
    )

    # Menampilkan konten berdasarkan pilihan menu
    if menu_selection == "Sistem Deteksi Kantuk":
        explain_drowsiness_detection()
    elif menu_selection == "Fitur Deteksi Kantuk":
        explain_drowsiness_detection_features()
    elif menu_selection == "Apa itu VGGFace?":
        explain_vggface_tab()
    elif menu_selection == "Apa itu EAR?":
        explain_ear_tab()
    elif menu_selection == "Cara Menggunakan":
        st.markdown(
            """
            ### Cara Menggunakan Aplikasi:
            1. Pilih **Drowsiness Detection** yang berada di bagian menu.
            2. Pilih fitur **Webcam** atau **Upload Gambar** untuk mendeteksi kantuk. 
            3. Pada fitur Webcam pilih **Start Detection** untuk memulai deteksi menggunakan webcam.
            4. Mata pengguna akan dipantau menggunakan algoritma EAR.
            5. Jika mata tertutup selama waktu tertentu, alarm akan berbunyi sebagai tanda peringatan.
            """
        )
