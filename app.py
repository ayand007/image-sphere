import streamlit as st
import numpy as np
import io

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from utils import load_image, to_image
import filters as f
from charts import plot_histogram, plot_rgb_histogram
from history import History
import adjustments as adj

def crop_image(arr, x1, x2, y1, y2):
    return arr[y1:y2, x1:x2].copy()

def extract_crop_coords(canvas_result, orig_w, orig_h, disp_w, disp_h):

    if canvas_result is None or canvas_result.json_data is None:
        return None
    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None

    rect = objects[-1]

    left = float(rect.get("left", 0))
    top = float(rect.get("top", 0))

    rect_w = float(rect.get("width", 0)) * abs(
        float(rect.get("scaleX", 1))
    )

    rect_h = float(rect.get("height", 0)) * abs(
        float(rect.get("scaleY", 1))
    )

    origin_x = rect.get("originX", "left")
    origin_y = rect.get("originY", "top")

    if origin_x == "center":
        left -= rect_w / 2
    elif origin_x == "right":
        left -= rect_w

    if origin_y == "center":
        top -= rect_h / 2
    elif origin_y == "bottom":
        top -= rect_h

    canvas_x1 = left
    canvas_y1 = top
    canvas_x2 = left + rect_w
    canvas_y2 = top + rect_h

    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    x1 = int(np.floor(canvas_x1 * scale_x))
    y1 = int(np.floor(canvas_y1 * scale_y))

    x2 = int(np.ceil(canvas_x2 * scale_x))
    y2 = int(np.ceil(canvas_y2 * scale_y))

    x1 = max(0, min(x1, orig_w))
    x2 = max(0, min(x2, orig_w))

    y1 = max(0, min(y1, orig_h))
    y2 = max(0, min(y2, orig_h))

    return x1, x2, y1, y2

st.set_page_config(
    page_title="Image Lab",
    layout="wide"
)


if "history" not in st.session_state:
    st.session_state.history = History()

if "canvas_revision" not in st.session_state:
    st.session_state.canvas_revision = 0

history = st.session_state.history

st.title("NumPy Image Lab")

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "bmp"]
)


if uploaded:

    upload_key = (
        uploaded.name,
        uploaded.size,
        getattr(uploaded, "file_id", None)
    )

    if st.session_state.get("last_uploaded") != upload_key:

        arr, original_img, is_rgb = load_image(uploaded)

        history.states = []
        history.index = -1

        history.push(arr)

        st.session_state["original_img"] = original_img
        st.session_state["is_rgb"] = is_rgb
        st.session_state["last_uploaded"] = upload_key

        st.session_state["_reset_editor_controls"] = True
        st.session_state.canvas_revision += 1


    original_img = st.session_state["original_img"]
    is_rgb = st.session_state["is_rgb"]

    img = history.current()

    if st.session_state.pop("_reset_editor_controls", False):

        h, w = img.shape[:2]

        defaults = {
            "brightness_val": 0,
            "contrast_val": 1.0,
            "threshold_val": 0,

            "color_filter_name": "None",
            "filter_strength": 1.0,

            "effect": "None",

            "rotate": 0,
            "flip_h": False,
            "flip_v": False,

            "use_roi": False,

            "roi_x1": 0,
            "roi_x2": w,
            "roi_y1": 0,
            "roi_y2": h,

            "enable_crop_tool": False,

            # Advanced
            "adv_temperature": 0,
            "adv_tint": 0,
            "adv_exposure": 0.0,
            "adv_highlights": 0,
            "adv_shadows": 0,
            "adv_whites": 0,
            "adv_blacks": 0,
            "adv_texture": 0,
            "adv_clarity": 0,
            "adv_dehaze": 0,
            "adv_vibrance": 0,
            "adv_saturation": 0,
            
            # Tone Curve
            "adv_tone_preset": "None",
            "adv_custom_curve": False,
            "adv_curve_black": 0,
            "adv_curve_shadows": 64,
            "adv_curve_midtones": 128,
            "adv_curve_highlights": 192,
            "adv_curve_whites": 255,
            
            # Color Grading
            "adv_blending": 50,
            "adv_balance": 0,
        }

        # HSL defaults
        for color in (
            "red", "orange", "yellow", "green",
            "aqua", "blue", "purple", "magenta"
        ):
            defaults[f"adv_hsl_{color}_h"] = 0
            defaults[f"adv_hsl_{color}_s"] = 0
            defaults[f"adv_hsl_{color}_l"] = 0
        
        
        # Three-way grading defaults
        for zone in ("shadows", "midtones", "highlights"):
            defaults[f"adv_{zone}_hue"] = 0
            defaults[f"adv_{zone}_saturation"] = 0
            defaults[f"adv_{zone}_luminance"] = 0
        st.session_state.update(defaults)

    st.sidebar.header(" Controls")

    brightness_val = st.sidebar.slider(
        "Brightness",
        -100, 100, 0,
        key="brightness_val"
    )

    contrast_val = st.sidebar.slider(
        "Contrast",
        0.5, 3.0, 1.0, 0.1,
        key="contrast_val"
    )

    threshold_val = st.sidebar.slider(
        "Threshold",
        0, 255, 0,
        key="threshold_val"
    )

    st.sidebar.markdown("###  Color Filters")

    color_filter_name = st.sidebar.selectbox(
        "Choose Color Filter",
        [
            "None",
            "Frblair",
            "Cinematic",
            "Vintage Film",
            "Lofi Dusk",
            "Dreamy Glow",
            "Teal & Orange"
        ],
        key="color_filter_name"
    )

    filter_strength = st.sidebar.slider(
        "Filter Strength",
        0.0, 1.0, 1.0, 0.05,
        key="filter_strength"
    )

    effect = st.sidebar.selectbox(
        "Filter",
        [
            "None",
            "Negative",
            "Blur",
            "Sharpen",
            "Edge Detect"
        ],
        key="effect"
    )

    rotate = st.sidebar.selectbox(
        "Rotate",
        [0, 90, 180, 270],
        key="rotate"
    )

    flip_h = st.sidebar.checkbox(
        "Flip Horizontal",
        key="flip_h"
    )

    flip_v = st.sidebar.checkbox(
        "Flip Vertical",
        key="flip_v"
    )

    st.sidebar.markdown("###  ROI (Effects only)")

    use_roi = st.sidebar.checkbox(
        "Enable ROI",
        key="use_roi"
    )

    if use_roi:

        h, w = img.shape[:2]

        x1 = st.sidebar.slider(
            "X1", 0, w - 1, 0,
            key="roi_x1"
        )

        x2 = st.sidebar.slider(
            "X2", 0, w, w,
            key="roi_x2"
        )

        y1 = st.sidebar.slider(
            "Y1", 0, h - 1, 0,
            key="roi_y1"
        )

        y2 = st.sidebar.slider(
            "Y2", 0, h, h,
            key="roi_y2"
        )

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])


    show_hist = st.sidebar.checkbox("Show Histogram")

    enable_crop_tool = st.sidebar.checkbox(
        "Enable Crop Tool",
        key="enable_crop_tool"
    )
    
    with st.sidebar.expander("Advanced Develop", expanded=False):
        st.markdown("#### Basic Adjustments")
    
        adv_temperature = st.slider(
            "Temperature", -100, 100, 0,
            key="adv_temperature"
        )
    
        adv_tint = st.slider(
            "Tint", -100, 100, 0,
            key="adv_tint"
        )
    
        adv_exposure = st.slider(
            "Exposure (EV)", -5.0, 5.0, 0.0, 0.1,
            key="adv_exposure"
        )
    
        adv_highlights = st.slider(
            "Highlights", -100, 100, 0,
            key="adv_highlights"
        )
    
        adv_shadows = st.slider(
            "Shadows", -100, 100, 0,
            key="adv_shadows"
        )
    
        adv_whites = st.slider(
            "Whites", -100, 100, 0,
            key="adv_whites"
        )
    
        adv_blacks = st.slider(
            "Blacks", -100, 100, 0,
            key="adv_blacks"
        )
    
        adv_texture = st.slider(
            "Texture", -100, 100, 0,
            key="adv_texture"
        )
    
        adv_clarity = st.slider(
            "Clarity", -100, 100, 0,
            key="adv_clarity"
        )
    
        adv_dehaze = st.slider(
            "Dehaze", -100, 100, 0,
            key="adv_dehaze"
        )
    
        adv_vibrance = st.slider(
            "Vibrance", -100, 100, 0,
            key="adv_vibrance"
        )
    
        adv_saturation = st.slider(
            "Saturation", -100, 100, 0,
            key="adv_saturation"
        )

        st.markdown("---")
        st.markdown("#### Tone Curve")
    
        adv_tone_preset = st.selectbox(
            "Curve Preset",
            [
                "None",
                "Linear",
                "Subtle S",
                "Strong S",
                "Matte",
                "Faded"
            ],
            key="adv_tone_preset"
        )
    
        adv_custom_curve = st.checkbox(
            "Enable Custom Curve",
            key="adv_custom_curve"
        )
    
        tone_curve_points = None
    
        if adv_custom_curve:
    
            st.caption("Adjust the output values of five fixed input points.")
    
            adv_curve_black = st.slider(
                "Black Point Output", 0, 255,
                key="adv_curve_black"
            )
    
            adv_curve_shadows = st.slider(
                "Shadows Output", 0, 255,
                key="adv_curve_shadows"
            )
    
            adv_curve_midtones = st.slider(
                "Midtones Output", 0, 255,
                key="adv_curve_midtones"
            )
    
            adv_curve_highlights = st.slider(
                "Highlights Output", 0, 255,
                key="adv_curve_highlights"
            )
    
            adv_curve_whites = st.slider(
                "White Point Output", 0, 255,
                key="adv_curve_whites"
            )
    
            tone_curve_points = [
                (0, adv_curve_black),
                (64, adv_curve_shadows),
                (128, adv_curve_midtones),
                (192, adv_curve_highlights),
                (255, adv_curve_whites)
            ]
    
        else:
    
            curve_mapping = {
                "Linear": "linear",
                "Subtle S": "subtle_s",
                "Strong S": "strong_s",
                "Matte": "matte",
                "Faded": "faded"
            }
    
            if adv_tone_preset != "None":
                tone_curve_points = adj.get_tone_curve_preset(
                    curve_mapping[adv_tone_preset]
                )
    
        st.markdown("---")
        st.markdown("#### HSL / Color Mix")
    
        if not is_rgb:
            st.caption("HSL adjustments require an RGB image.")
    
        hsl_adjustments = {}
    
        colors = [
            "red", "orange", "yellow", "green",
            "aqua", "blue", "purple", "magenta"
        ]
    
        for color in colors:
    
            st.markdown(f"**{color.title()}**")
    
            hue_col, sat_col, lum_col = st.columns(3)
    
            hsl_adjustments[color] = {
                "h": hue_col.slider(
                    "H", -100, 100, 0,
                    key=f"adv_hsl_{color}_h",
                    disabled=not is_rgb
                ),
    
                "s": sat_col.slider(
                    "S", -100, 100, 0,
                    key=f"adv_hsl_{color}_s",
                    disabled=not is_rgb
                ),
    
                "l": lum_col.slider(
                    "L", -100, 100, 0,
                    key=f"adv_hsl_{color}_l",
                    disabled=not is_rgb
                )
            }

        st.markdown("---")
        st.markdown("#### 3-Way Color Grading")
    
        color_grading = {}
    
        for zone in ("shadows", "midtones", "highlights"):
    
            st.markdown(f"**{zone.title()}**")
    
            hue_col, sat_col, lum_col = st.columns(3)
    
            color_grading[zone] = {
                "hue": hue_col.slider(
                    "Hue", 0, 360, 0,
                    key=f"adv_{zone}_hue",
                    disabled=not is_rgb
                ),
    
                "saturation": sat_col.slider(
                    "Sat", 0, 100, 0,
                    key=f"adv_{zone}_saturation",
                    disabled=not is_rgb
                ),
    
                "luminance": lum_col.slider(
                    "Lum", -100, 100, 0,
                    key=f"adv_{zone}_luminance",
                    disabled=not is_rgb
                )
            }
    
        adv_blending = st.slider(
            "Blending", 0, 100, 50,
            key="adv_blending",
            disabled=not is_rgb
        )
    
        adv_balance = st.slider(
            "Balance", -100, 100, 0,
            key="adv_balance",
            disabled=not is_rgb
        )

    def process_channel(channel):

        if brightness_val != 0:
            channel = f.brightness(channel, brightness_val)
        if contrast_val != 1.0:
            channel = f.contrast(channel, contrast_val)
        if threshold_val > 0:
            channel = f.threshold(channel, threshold_val)
        if effect == "Negative":
            channel = f.negative(channel)
        elif effect == "Blur":
            channel = f.blur(channel)
        elif effect == "Sharpen":
            channel = f.sharpen(channel)
        elif effect == "Edge Detect":
            channel = f.edge_detect(channel)
        return channel


    channel_processing_needed = (
        brightness_val != 0
        or contrast_val != 1.0
        or threshold_val > 0
        or effect != "None"
    )


    if is_rgb:

        if channel_processing_needed:
            processed = np.stack(
                [
                    process_channel(img[:, :, i])
                    for i in range(3)
                ],
                axis=2
            )

        else:
            processed = img

        processed = f.color_filter(
            processed,
            color_filter_name,
            filter_strength
        )

    else:

        processed = (
            process_channel(img)
            if channel_processing_needed
            else img
        )

        if color_filter_name != "None":
            st.info("Color filters are available for RGB images.")

    advanced_basic = {
        "temperature": adv_temperature,
        "tint": adv_tint,
        "exposure": adv_exposure,
        "highlights": adv_highlights,
        "shadows": adv_shadows,
        "whites": adv_whites,
        "blacks": adv_blacks,
        "texture": adv_texture,
        "clarity": adv_clarity,
        "dehaze": adv_dehaze,
        "vibrance": adv_vibrance,
        "saturation": adv_saturation
    }

    basic_to_apply = (
        advanced_basic
        if any(value != 0 for value in advanced_basic.values())
        else None
    )

    hsl_to_apply = {
        color: values
        for color, values in hsl_adjustments.items()
        if any(value != 0 for value in values.values())
    } if is_rgb else None


    grading_to_apply = {
        zone: values
        for zone, values in color_grading.items()
        if any(value != 0 for value in values.values())
    } if is_rgb else None

    if grading_to_apply:

        grading_to_apply["blending"] = adv_blending
        grading_to_apply["balance"] = adv_balance

    if (
        basic_to_apply is not None
        or tone_curve_points is not None
        or hsl_to_apply
        or grading_to_apply
    ):

        processed = adj.develop_image(
            processed,
            basic=basic_to_apply,
            curve_points=tone_curve_points,
            hsl=hsl_to_apply,
            color_grading=grading_to_apply
        )


    if use_roi:
        result = img.copy()
        if is_rgb:

            result[y1:y2, x1:x2, :] = (
                processed[y1:y2, x1:x2, :]
            )

        else:
            result[y1:y2, x1:x2] = (
                processed[y1:y2, x1:x2]
            )
    else:
        result = processed

    if rotate != 0:
        result = np.rot90(result, rotate // 90)

    if flip_h:
        result = result[:, ::-1]

    if flip_v:
        result = result[::-1, :]

    orig_h, orig_w = result.shape[:2]

    MAX_CANVAS_WIDTH = 840
    MAX_CANVAS_HEIGHT = 1080

    scale = min(
        1.0,
        MAX_CANVAS_WIDTH / orig_w,
        MAX_CANVAS_HEIGHT / orig_h
    )

    disp_w = max(1, int(round(orig_w * scale)))
    disp_h = max(1, int(round(orig_h * scale)))

    edited_display = to_image(result).resize(
        (disp_w, disp_h),
        resample=Image.Resampling.LANCZOS
    )


    colA, colB = st.columns(2)

    with colA:

        st.subheader("Original")
        st.image(original_img)

    with colB:

        st.subheader("Preview / Edited")
        st.image(edited_display)

    crop_coords = None

    if enable_crop_tool:

        st.markdown("###  Crop Tool")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.15)",
            stroke_width=2,
            stroke_color="#FF3333",

            background_image=edited_display,

            height=disp_h,
            width=disp_w,

            drawing_mode="rect",

            update_streamlit=True,

            key=f"crop_canvas_{st.session_state.canvas_revision}"
        )

        crop_coords = extract_crop_coords(
            canvas_result,
            orig_w,
            orig_h,
            disp_w,
            disp_h
        )

    col1, col2, col3, col4 = st.columns(4)

    if col1.button(" Save Changes"):

        saved = history.push(result)

        if saved:

            st.session_state["_reset_editor_controls"] = True
            st.session_state.canvas_revision += 1

            st.rerun()

        else:

            st.info("No new changes to save.")

    if col2.button(
        " Crop Selected Area",
        disabled=not enable_crop_tool
    ):

        if crop_coords is not None:

            x1, x2, y1, y2 = crop_coords

            if (x2 - x1) > 5 and (y2 - y1) > 5:

                cropped = crop_image(
                    result,
                    x1, x2, y1, y2
                )

                # Commit the CROPPED image directly.
                # Do not push the uncropped image here.
                saved = history.push(cropped)

                if saved:

                    st.session_state["_reset_editor_controls"] = True
                    st.session_state.canvas_revision += 1

                    st.rerun()

                else:

                    st.info("This crop does not create a new image state.")

            else:

                st.warning("Selected area is too small.")

        else:

            st.warning("Draw a rectangle first.")

    if col3.button("↩ Undo"):

        history.undo()

        st.session_state["_reset_editor_controls"] = True
        st.session_state.canvas_revision += 1

        st.rerun()

    if col4.button("↪ Redo"):

        history.redo()

        st.session_state["_reset_editor_controls"] = True
        st.session_state.canvas_revision += 1

        st.rerun()
    if show_hist:

        st.markdown("---")

        if is_rgb:
            plot_rgb_histogram(result, "RGB Histogram")

        else:
            plot_histogram(result, "Histogram")

    buffer = io.BytesIO()

    final_img = to_image(result)

    final_img.save(buffer, format="PNG")

    st.download_button(
        "⬇ Download Image",
        data=buffer.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )


else:

    st.info("Upload an image to start.")
