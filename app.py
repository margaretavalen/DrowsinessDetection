import streamlit as st
from streamlit_option_menu import option_menu
from home import home_page
from drowsiness_detection import drowsiness_detection_page
from about import about_page

# Streamlit UI - Page Navigation
def main():
    st.set_page_config(page_title="Drowsiness Detection System", layout="wide")

    # Sidebar Navigation using option_menu
    with st.sidebar:
        page = option_menu(
            "Menu",
            ["Home", "Drowsiness Detection", "About"],
            icons=["house", "eye", "info-circle"],  
            menu_icon="cast",  # Ikon menu utama
            default_index=0,  
        )

    # Halaman yang ditampilkan
    if page == "Home":
        home_page()
    elif page == "Drowsiness Detection":
        drowsiness_detection_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
