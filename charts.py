import matplotlib.pyplot as plt
import streamlit as st

def plot_histogram(arr,title):
    fig,axes = plt.subplots(figsize=(6,3))
    axes.hist(arr.ravel(), bins = 50)
    axes.set_title(title)
    st.pyplot(fig)

def plot_rgb_histogram(arr, title):
    fig, axes = plt.subplots(figsize=(6,3))

    colors = ['red', 'green', 'blue']

    for i, color in enumerate(colors):
        axes.hist(arr[:,:,i].ravel(), bins=50, color=color, alpha=0.5)

    axes.set_title(title)
    st.pyplot(fig)