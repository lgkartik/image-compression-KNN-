import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from skimage import io
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from plot_utils import plot_utils
def show_image(filename):
    img = io.imread(filename)
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img);
    st.pyplot()
def show_colormap_orig(filename):
    img = io.imread(filename)
    img_data = (img/255.0).reshape(-1,3)
    x = plot_utils(img_data,title="Input color space")
    x.colorSpace()
    st.pyplot()

def show_compressed_colormap(filename):
    img = io.imread(filename)
    img_data = (img/255.0).reshape(-1,3)
    kmeans = MiniBatchKMeans(16).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
    y = plot_utils(img_data, colors=k_colors, title="Reduced color space: 16 colors")
    y.colorSpace()
    st.pyplot()

def color_compression(image,k):
    input_img = io.imread(image)
    img_data = (input_img / 255.0).reshape(-1, 3)
    kmeans = MiniBatchKMeans(k).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
    #After K-means has converged, load the large image into your program and
    #replace each of its pixels with the nearest of the centroid colors you found
    #from the small image.
    k_img = np.reshape(k_colors, (input_img.shape))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('K-means Image Compression', fontsize=20)

    ax1.set_title('Compressed')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)

    ax2.set_title('Original (16,777,216 colors)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(input_img)

    plt.subplots_adjust(top=0.85)
    plt.show()
    st.pyplot()

st.title("Image compression using KNN")
st.sidebar.title("Upload image")
filename = st.sidebar.file_uploader("Choose file")
if filename is not None:
    if st.sidebar.checkbox("Show uploaded image",False):
        show_image(filename)
    if st.sidebar.checkbox("Show original color space",False):
        show_colormap_orig(filename)
    if st.sidebar.checkbox("Show compressed color space",False):
        show_compressed_colormap(filename)
    if st.sidebar.checkbox("compare compressed with real images",False):
        k =st.slider('Select k value', 1, 50, 1)
        color_compression(filename,k)
