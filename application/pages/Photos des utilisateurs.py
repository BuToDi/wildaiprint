import streamlit as st

with st.container():


    collogo, colnom = st.columns([0.5, 3])

    with collogo:
        st.image("assets/logo_vert.webp", width=100)

    with colnom: 
        st.markdown(
    "<p style='font-size:50px; color:#31C48D;'>WILDAIPRINT</p>",
    unsafe_allow_html=True)

st.markdown(
    """<hr style="border: none; height: 2px; background-color: #31C48D;" />""",
    unsafe_allow_html=True
)