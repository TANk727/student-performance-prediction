
import streamlit as st


def apply_custom_theme():

    st.markdown(
        """
        <style>

        /* Main application background */
        .stApp {
            background-color: #0e1117;
            color: #f0f2f6;
        }

        /* Main content spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }

        /* Headings */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -1px;
        }

        h2, h3 {
            font-weight: 600 !important;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 12px;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 700;
        }

        /* Buttons */
        .stButton button,
        .stFormSubmitButton button {
            border-radius: 8px;
            font-weight: 600;
            min-height: 42px;
        }

        /* Select boxes and input fields */
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 8px;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
        }

        /* Dividers */
        hr {
            border-color: #30363d;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid #30363d;
            border-radius: 10px;
            overflow: hidden;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def page_header(title, subtitle):

    st.title(title)

    st.markdown(
        f"<p style='color:#9ca3af; font-size:16px;'>{subtitle}</p>",
        unsafe_allow_html=True
    )
