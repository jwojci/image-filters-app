import io
import base64
import cv2
import numpy as np
from PIL import Image
from filters import *
import streamlit as st

filter_funcs = {"None": None,
                "Black and White": bw_filter,
                "Sepia / Vintage": sepia,
                "Vignette Effect": vignette,
                "Pencil Sketch": pencil_sketch,
                "Oil Painting": oil_painting,
                "Invert": invert,
                }

filter_imgs = {"None": "filter_none.png",
               "Black and White": "filter_bw.png",
               "Sepia / Vintage": "filter_sepia.png",
               "Vignette Effect": "filter_vignette.png",
               "Pencil Sketch": "filter_pencil_sketch.png",
               "Oil Painting": "filter_oil_painting.png",
               "Invert": "filter_invert.png"
               }


# Generating a link to download a particular image file
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Set title
st.title("Artistic Image Filters")

# Upload image
uploaded_file = st.file_uploader("Choose an image file:", type=["png", "jpg"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.header("Original")
        # Display uploaded image
        st.image(img, channels="BGR", use_column_width=True)

    st.header("Filter Examples:")
    # Display a selection box for choosing the filter to apply
    option = st.selectbox("Select a filter:", list(filter_funcs.keys()))

    # Define columns for thumbnail images
    cols = st.columns(4)
    for i, key in enumerate(filter_funcs.keys()):
        col = cols[i % 4]
        with col:
            st.caption(key)
            st.image(f"{filter_imgs[key]}")

    # Flag for showing output image
    output_flag = 1
    # Colorspace of output image
    color = "BGR"

    # Generate a filtered image based on the selected option
    if option == "None":
        # Don't show output image
        output_flag = 0
    elif option == "Black and White":
        output = bw_filter(img)
        color = "GRAY"
    elif option == "Sepia / Vintage":
        output = sepia(img)
    elif option == "Vignette Effect":
        level = st.slider("level", 0, 5, 2)
        output = vignette(img, level)
    elif option == "Pencil Sketch":
        ksize = st.slider("Blur kernel size", 1, 11, 5, step=2)
        output = pencil_sketch(img, ksize)
        color = "GRAY"
    elif option == "Oil Painting":
        output = oil_painting(img)
    elif option == "Invert":
        output = invert(img)

    with output_col:
        if output_flag == 1:
            st.header("Output")
            st.image(output, channels=color)
            # fromarray convert cv2 image into PIL format for saving it using download link
            if color == "BGR":
                result = Image.fromarray(output[:, :, ::-1])
            else:
                result = Image.fromarray(output)
            # Display link
            st.markdown(get_image_download_link(result, "output.png", "Download " + "Output"), unsafe_allow_html=True)
