import streamlit as st
from streamlit_option_menu import option_menu

# 🌐 Page Config
st.set_page_config(page_title="EMI Prediction App", page_icon="💳", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.markdown("<h4 style='margin-bottom:10px;font-size:24px'>📌 Menu</h4>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Eligibility Predictor",
            "EMI Predictor",
            # "MLflow Dashboard",
            # "Data Explorer",
            # "Admin Panel"   # ✅ Added new menu
        ],
        icons=[
            "house", 
            "check-circle", 
            "currency-dollar", 
            "bar-chart", 
            "database",
            # "gear"   # ✅ Gear icon for admin
        ],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
                "white-space": "nowrap",
            },
            "nav-link-selected": {
                "background-color": "#4a66d5",
                "color": "white",
                "font-size": "15px"
            },
        }
    )

# Load page dynamically
if selected == "Home":
    from views import home
    home.render()

elif selected == "Eligibility Predictor":
    from views import classification
    classification.render()

elif selected == "EMI Predictor":
    from views import regression
    regression.render()

elif selected == "MLflow Dashboard":
    from views import mlflow_dashboard
    mlflow_dashboard.render()

elif selected == "Data Explorer":
    from views import data_explorer
    data_explorer.render()

# elif selected == "Admin Panel":   # ✅ New menu routing
#     from views import admin
#     admin.render()