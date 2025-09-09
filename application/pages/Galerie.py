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


st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)

st.title("Galerie")


col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="center")

with col1:
    st.image("assets/chat.webp")
    st.image("assets/castor.webp")

with col2: 
    st.image("assets/lapin.webp")
    st.image("assets/ecureil.webp")

with col3:
    st.image("assets/chat.webp")
    st.image("assets/chat.webp")