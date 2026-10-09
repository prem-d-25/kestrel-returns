import streamlit as st

import batch_tab
import evidence_tab
import order_tab

st.set_page_config(page_title="Kestrel Returns Portal", page_icon="🦅", layout="wide")

logo_col, title_col = st.columns([1, 12])
with logo_col:
    st.image("app/assets/logo.svg", width=70)
with title_col:
    st.title("Kestrel Returns Portal")

tab1, tab2, tab3 = st.tabs(["Batch Predictions", "Check an Order", "Model Evidence"])
with tab1:
    batch_tab.show()
with tab2:
    order_tab.show()
with tab3:
    evidence_tab.show()