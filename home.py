import streamlit as st

def home_page():
    # Header dengan format
    st.markdown(
        """
        <h1 style="text-align: center; color: #ffffff; font-size: 40px; font-weight: bold;">
        Drowsiness Detection System
        </h1>
        """,
        unsafe_allow_html=True,
    )

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
        """,
        unsafe_allow_html=True,
    )

    # Gambar dengan caption
    st.image('images/drowsiness.jpg', caption='Drowsiness Detection System', use_container_width=True)
