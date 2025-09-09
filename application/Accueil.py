import streamlit as st
import pathlib
import os
import streamlit as st
import datetime
from PIL import Image, ExifTags
import tensorflow as tf
from ipyleaflet import Map, Marker
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import pandas as pd
from streamlit_carousel import carousel
import streamlit as st




df = pd.read_csv("application/dataphoto2.csv")


with st.container():


    collogo, colnom = st.columns([0.5, 3])

    with collogo:
        st.image("application/assets/logo_vert.webp", width=100)

    with colnom: 
        st.markdown(
    "<p style='font-size:50px; color:#31C48D;'>WILDAIPRINT</p>",
    unsafe_allow_html=True)

st.markdown(
    """<hr style="border: none; height: 2px; background-color: #31C48D;" />""",
    unsafe_allow_html=True
)

st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)



if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

def login():
    st.session_state.is_logged_in = True

def logout():
    st.session_state.is_logged_in = False
    st.session_state.user_name = ""

if not st.session_state.is_logged_in:
    st.button("Log in", on_click=login)
    st.session_state.user_name = st.text_input("Nom d'utilisateur", "")
    st.stop()

st.markdown(f"Bienvenue {st.session_state.user_name} !")
st.button("Log out", on_click=logout)

# stocker "st.session_state.user_name" dans un df avec l'heure et la date


with st.container():
    st.header("Bonjour")

train_ds = tf.keras.utils.image_dataset_from_directory(
    r"application/Mammiferes",        # chemin vers ton dossier
    validation_split=0.2,       # 20% sera réservé à la validation
    subset="training",
    seed=123,                   # pour reproductibilité
    image_size=(224, 224),      # redimensionne toutes les images
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    r"application/Mammiferes",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)


class_names= df['Espèce'] 



normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))


print("Classes détectées :", class_names)

for images, labels in train_ds.take(1):
    print("Shape d’un batch :", images.shape)
    print("Labels :", labels.numpy())


classifier = Sequential() 


classifier.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten()) 
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dense(units=12, activation='softmax'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


def load_model() :
    return classifier


def try_model():    
    with st.spinner("Loading Model"):
        classifier = load_model()
    
img_file_buffer = st.camera_input("Découvrez un animal à travers ses empreintes.")
    
if img_file_buffer is not None:
    # To read image file buffer as a 3D uint8 tensor with TensorFlow:
    bytes_data = img_file_buffer.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=3)

        # Check the type of img_tensor:
        # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
    st.write(f"Taille de l'image : {img_tensor.shape}")

        # Check the shape of img_tensor:
        # Should output shape: (height, width, channels)
    st.write(img_tensor.shape)



if img_file_buffer is not None:
    # Redimensionner à 64x64 car le modèle attend cette taille
    image = Image.open(img_file_buffer).resize((224, 224))
    img_array = np.array(image) / 255.0   # normalisation
    img_array = np.expand_dims(img_array, axis=0)  # batch size = 1

    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Prédiction"):
        pred = classifier.predict(img_array)
        jour = datetime.datetime.now().strftime('%d-%m-%Y')
        heure =datetime.datetime.now().strftime("%H:%M")
        predicted_class = class_names[np.argmax(pred)]
        animal_info = df[df["Espèce"] == predicted_class]
        info = animal_info.iloc[0]
        st.title(" Mais qui est passé par ici ?")
        st.markdown(f"Vous avez marché sur les traces d'un **{predicted_class}**")
        #st.markdown(f"Photo prise le **{jour}** à **{heure}** ")
        st.markdown(f"""
        ### 🧾 Informations sur cet animal indomptable

        - **Espèce :** {info['Espèce']}
        - **Nom latin :** {info['Nom latin']}
        - **Famille :** {info['Famille']}
        - **Description :** {info['Description']}
        - **Taille :** {info['Taille']}
        - **Région :** {info['Région']}
        - **Habitat :** {info['Habitat']}
        - **Fun Fact :** {info['Fun fact']}
        """)
        st.image(info["Photo"], use_container_width=True)
        st.image(info["Empreintes"], use_container_width=True)
        if predicted_class == "Ours":
            save_path = r"application/Mammiferes/Ours"
        elif predicted_class == "Loup":
            save_path = r"application/Mammiferes/Loup"
        elif predicted_class == "Rat":
            save_path = r"application/Mammiferes/Rat"
        elif predicted_class == "Raton laveur":
            save_path = r"application/Mammiferes/Raton laveur"
        elif predicted_class == "Puma":
            save_path = r"application/Mammiferes/Puma"
        elif predicted_class == "Renard":
            save_path = r"application/Mammiferes/Renard"
        elif predicted_class == "Lynx":
            save_path = r"application/Mammiferes/Lynx"
        elif predicted_class == "Lapin":
            save_path = r"application/Mammiferes/Lapin"
        elif predicted_class == "Ecureuil":
            save_path = r"application/Mammiferes/Ecureuil"
        elif predicted_class == "Coyote":
            save_path = r"application/Mammiferes/Coyote"
        elif predicted_class == "Chien":
            save_path = r"application/Mammiferes/Chien"
        elif predicted_class == "Chat":
            save_path = r"application/Mammiferes/Chat"
        elif predicted_class == "Castor":
            save_path = r"application/Mammiferes/Castor"
       
        file_path = os.path.join(save_path, img_file_buffer.name)

        with open(file_path,"wb") as f :
            f.write(img_file_buffer.getbuffer())



def try_model():    
    with st.spinner("Loading Model"):
        classifier = load_model()
        jour = datetime.datetime.now().strftime('%d-%m-%Y')
        heure =datetime.datetime.now().strftime("%H:%M")


uploaded_file = st.file_uploader("Ou importez une photo.", type=["jpg", "jpeg", "png"])







if uploaded_file is not None:
        # Redimensionner à 64x64 car le modèle attend cette taille
        image = Image.open(uploaded_file).resize((224, 224))
        img_array = np.array(image) / 255.0   # normalisation
        img_array = np.expand_dims(img_array, axis=0)  # batch size = 1

        #st.image(image, caption=f"Photo prise le **{jour}** à **{heure}** ", width=300)

        with st.spinner("Predicting"):
            pred = classifier.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]
        animal_info = df[df["Espèce"] == predicted_class]
        info = animal_info.iloc[0]
        st.title("🤌 Mate la taille du panard !")
        st.markdown(f"Vous avez marché sur les traces d'un **{predicted_class}**, c'est sur à 5%")
        st.image(image, caption="Uploaded Image", width=300)

        
        st.markdown(f"""
        ### 🧾 Informations sur cet animal indomptable

        - **Espèce :** {info['Espèce']}
        - **Nom latin :** {info['Nom latin']}
        - **Famille :** {info['Famille']}
        - **Description :** {info['Description']}
        - **Taille :** {info['Taille']}
        - **Région :** {info['Région']}
        - **Habitat :** {info['Habitat']}
        - **Fun Fact :** {info['Fun fact']}
        """)
        st.image(info["Photo"], use_container_width=True)
        st.image(info["Empreintes"], use_container_width=True)
        if predicted_class == "Ours":
            save_path = r"application/Mammiferes/Ours"
        elif predicted_class == "Loup":
            save_path = r"application/Mammiferes/Loup"
        elif predicted_class == "Rat":
            save_path = r"application/Mammiferes/Rat"
        elif predicted_class == "Raton laveur":
            save_path = r"application/Mammiferes/Raton laveur"
        elif predicted_class == "Puma":
            save_path = r"application/Mammiferes/Puma"
        elif predicted_class == "Renard":
            save_path = r"application/Mammiferes/Renard"
        elif predicted_class == "Lynx":
            save_path = r"application/Mammiferes/Lynx"
        elif predicted_class == "Lapin":
            save_path = r"application/Mammiferes/Lapin"
        elif predicted_class == "Ecureuil":
            save_path = r"application/Mammiferes/Ecureuil"
        elif predicted_class == "Coyote":
            save_path = r"application/Mammiferes/Coyote"
        elif predicted_class == "Chien":
            save_path = r"application/Mammiferes/Chien"
        elif predicted_class == "Chat":
            save_path = r"application/Mammiferes/Chat"
        elif predicted_class == "Castor":
            save_path = r"application/Mammiferes/Castor"
       
        file_path = os.path.join(save_path, uploaded_file.name)

        with open(file_path,"wb") as f :
            f.write(uploaded_file.getbuffer())
            


    
try_model()

st.markdown("<div style='margin-top: 90px;'></div>", unsafe_allow_html=True)




with st.container():
    
    st.markdown(
    """
    <a href="./Galerie" style="
        text-decoration: none;
        color: #31333f;
        font-size: 35px;
        font-weight: 500;
        ">
        Galerie
    </a>
    """,
    unsafe_allow_html=True
)
    
    from streamlit_carousel import carousel


    test_items = [
        dict(
            title="Chat",
            text="Un chat",
            img="assets/chat.webp",
        ),
        dict(
            title="Castor",
            text="Un castor",
            img="assets/castor.webp",
        ),
        dict(
            title="Lapin",
            text="Un lapin",
            img="assets/lapin.webp",
        ),
        dict(
            title="Ecureuil",
            text="Un écureuil",
            img="assets/ecureil.webp",
        ),
        dict(
            title="Chien",
            text="Un chien",
            img="assets/chien.webp",
        ),
    ]

    carousel(items=test_items)

    