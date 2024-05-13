import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

st.set_page_config(
    page_title="Plant-Disease-Detection",
    page_icon= "🌿",
    initial_sidebar_state = "auto"
)


# Tensorflow model prediction
def model_prediction(input_image):
    trained_model = tf.keras.models.load_model("model.keras")
    image = tf.keras.preprocessing.image.load_img(input_image,target_size=(228,228))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # To convert single image to batch
    predictions = trained_model.predict(input_arr)
    result_index = np.argmax(predictions)

    return result_index

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #bff2ca;
    }
</style>
""", unsafe_allow_html=True)

# sidebar
img_side = "Images/img2.png"
with st.sidebar:
    with st.container():
        l, m, r = st.columns((1,3,1))
        with l:
            st.empty()
        with m:
            st.image(img_side, width=175)
        with r:
            st.empty()
    
    choose = option_menu(
                        "Dashboard", 
                        ["Home","About","Disease Recognition"],
                         icons=['book half', 'globe',  'tools'],
                         menu_icon="plant", 
                         default_index=0,
                         styles={
        "container": {"padding": "0!important", "background-color": "#bff2ca"},
        "icon": {"color": "darkorange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"5px", "--hover-color": "#65eb82"},
        "nav-link-selected": {"background-color": "#65eb82"},
    }
    )


if choose == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = ("Images/plant.jpeg")
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission to help in identifying plant disease efficintly. Upload an image of plant, and our system will and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Got to the **Disease Recognition** page and upload an Image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advances algorithms to identify potential diseases.
    3. **Results:** View the results and recomendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fats and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on **Disease Recognition** page in the sidebar to upload and image and experience the power of our Plant Recognition System!

    ### About us
    Learn about the project, our team, and our goals on the **About** page.
    """)
    st.markdown("""

    ## Check out the Github repo here 👉
    # """)
    mention(label = "Github Repo",icon = "github", url = "https://github.com/aman977381/Plant-Disease-Recoginition.git")

elif choose == "About":
    st.header("About")
    st.markdown("""
    ### About dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    
    ### Content
    1. Train (70295 Images)
    2. Valid (17572 Images)
    3. Test (33 Images)


    """)

elif choose == "Disease Recognition":
    st.header("Disease Recognition")
    input_image = st.file_uploader("Choose an Image:",type=['jpg', 'png', 'jpeg'])
    if not input_image:
        input_image = "Images/test.jpg"
    if st.button("show image"):
        st.image(input_image, use_column_width=True)
    
    # Predicting Image
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(input_image)
        class_name = ['Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy']
        
        model_predicted = class_name[result_index]
        st.success("Model is Predicting it's a {}".format(model_predicted))
