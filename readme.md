# 📸 Image Sphere

**A NumPy-powered Image Editor built with Python and Streamlit.**

Image Sphere  is an interactive image editing application that uses Python and NumPy to perform image processing and manipulation operations. It combines fundamental digital image processing techniques with modern photo-editing features inspired by applications such as Adobe Lightroom and Instagram.

The project focuses on implementing image transformations, filters, color grading, and advanced adjustments using mathematical operations and NumPy arrays.

---

## 🚀 Features

### 1. Basic Image Adjustments

- **Brightness:** Adjust the overall brightness of an image.
- **Contrast:** Control the difference between light and dark regions.
- **Thresholding:** Convert images into binary representations based on pixel intensity.
- **Negative Transformation:** Invert pixel intensity values.

### 2. Image Filters

Apply different image-processing filters:

- **Blur:** Smooth image details using convolution.
- **Sharpen:** Enhance edges and fine details.
- **Edge Detection:** Detect prominent edges using convolution kernels.

### 3. Instagram-Style Color Filters

Enhance photographs with predefined color-filter styles:

- Frblair
- Cinematic
- Vintage Film
- Lofi Dusk
- Dreamy Glow
- Teal & Orange

A filter-strength slider allows users to control the intensity of the selected effect.

### 4. Advanced Develop – Lightroom-Inspired Editing

Image Sphere also includes an advanced photo-editing module.

**Basic Adjustments**
- Temperature
- Tint
- Exposure
- Highlights
- Shadows
- Whites
- Blacks
- Texture
- Clarity
- Dehaze
- Vibrance
- Saturation

**Tone Curve**
- Linear
- Subtle S-Curve
- Strong S-Curve
- Matte
- Faded
- Custom five-point tone curve

**HSL / Color Mix**

Independently adjust Hue, Saturation, and Luminance for:

- Red
- Orange
- Yellow
- Green
- Aqua
- Blue
- Purple
- Magenta

**Three-Way Color Grading**

Apply independent color grading to:

- Shadows
- Midtones
- Highlights

Additional controls:
- Hue
- Saturation
- Luminance
- Blending
- Balance

### 5. Region of Interest (ROI)

Select a specific region of an image and apply image-processing operations to that selected area.

This allows localized editing without affecting the entire image.

### 6. Crop Tool

- Interactive rectangle-based crop selection.
- Crop directly from the image preview.
- Preserve the original image resolution when cropping.

### 7. Geometric Transformations

- Rotate images by 0°, 90°, 180°, and 270°.
- Horizontal flipping.
- Vertical flipping.

### 8. Histogram Analysis

Analyze image intensity distributions using:

- Grayscale histogram.
- Individual RGB channel histograms.

### 9. Image History Management

- Undo previous saved changes.
- Redo reverted changes.
- Maintain editing states through a history system.

### 10. Image Export

Download the final edited image in PNG format.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Mathematical image processing and pixel manipulation |
| Streamlit | Interactive web application interface |
| Pillow (PIL) | Image loading, conversion, resizing, and export |
| Matplotlib | Histogram visualization |
| Streamlit Drawable Canvas | Interactive crop selection |

---

## 📂 Project Structure

```text
numpy-image-lab/
│
├── app.py              # Main Streamlit application
├── adjustments.py      # Advanced photo adjustments
├── filters.py          # Image filters and color effects
├── utils.py            # Image loading and conversion
├── charts.py           # Histogram visualization
├── history.py          # Undo and redo management
│
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignored files
└── README.md           # Project documentation
```

---

## 🧠 Image Processing Approach

The primary objective of this project is to explore how digital image processing operations can be implemented using mathematical concepts and NumPy.

Some of the techniques involved include:

- Array slicing and indexing.
- Pixel-wise arithmetic operations.
- RGB channel manipulation.
- Vectorized computations.
- Convolution and kernel operations.
- Luminance-based color transformations.
- Histogram analysis.
- Interpolation for tone curve adjustments.
- Region-based image processing.

Instead of depending entirely on ready-made image-filtering functions, the project implements many of its image transformations through NumPy operations.

---

## 🎯 Project Objectives

1. Understand the mathematical foundations of digital image processing.
2. Apply NumPy for practical image manipulation and transformation.
3. Explore color spaces, color grading, and tonal adjustments.
4. Develop an interactive image editor using Streamlit.
5. Implement modular and reusable image-processing functions.
6. Combine theoretical image-processing concepts with a practical software application.

---

## 👨‍💻 Author

**Ayan Dutta**
**Aratrika Das**

Developed as a practical Python and NumPy project exploring digital image processing, mathematical transformations, and interactive software development.

---

## ⭐ Acknowledgements

This project is inspired by modern photo-editing applications and the mathematical foundations of digital image processing.

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

---

**Built with Python, NumPy, and Streamlit.**