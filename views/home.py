import streamlit as st

def render():
    # 🌐 Custom CSS for styling
    st.markdown("""
        <style>
            /* Card Style */
            .card {
                background: #ffffff;
                padding: 25px;
                border-radius: 14px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                text-align: center;
                transition: all 0.3s ease-in-out;
                cursor: pointer;
            }
            .card:hover {
                transform: translateY(-8px);
                box-shadow: 0 8px 18px rgba(0,0,0,0.2);
            }
            .card h4 {
                margin-top: 10px;
                font-size: 20px;
                color: #333;
            }
            .card p {
                font-size: 15px;
                color: #555;
            }

            /* Admin Panel */
            .admin {
                background-color: #fffbea;
                padding: 20px;
                border-radius: 12px;
                border-left: 6px solid #ffc107;
                margin-top: 40px;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            .admin h4 {
                color: #856404;
            }
            /* Hero Banner */
            .hero {
                text-align:center;
                padding:50px 0 30px 0;
                background: linear-gradient(135deg, #42a5f5, #7e57c2);
                color: white;
                border-radius: 12px;
                margin-bottom: 40px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            .hero h1 {
                font-size: 50px;
                margin-bottom: 12px;
            }
            .hero p {
                font-size: 20px;
                opacity: 0.9;
            }
        </style>
    """, unsafe_allow_html=True)

    # 🎨 Hero Banner
    st.markdown("""
        <div class="hero">
            <h1>💳 EMI Prediction Platform</h1>
            <p>A smart, ML-powered tool to assess loan eligibility and EMI limits.</p>
        </div>
    """, unsafe_allow_html=True)

    # 🚀 Explore Modules
    st.subheader("🚀 Explore the Modules")

    col1, col2 = st.columns(2)

    with col1:        
        st.markdown("""
            <div class="card">
                <h4>🎯 Eligibility Predictor</h4>
                <p>Classify users as Eligible, High Risk and Not Eligible using ML models.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="card">
                <h4>💰 EMI Predictor</h4>
                <p>Estimate the maximum EMI a user can afford based on their profile.</p>
            </div>
        """, unsafe_allow_html=True)

    # 🛠️ Admin Panel
    # st.markdown("""
    #     <div class="admin">
    #         <h4>🛠️ Admin Panel</h4>
    #         <p>📂 Upload new datasets<br>📦 Manage saved models<br>🔄 Reset your workspace</p>
    #     </div>
    # """, unsafe_allow_html=True)

    # 📣 Footer
    st.markdown("""
        <div style="text-align:center;margin-top:40px;font-size:13px;color:#555;">
            <hr>
            Built with ❤️ by <b>Satheesh</b> • Powered by Streamlit & MLflow
        </div>
    """, unsafe_allow_html=True)