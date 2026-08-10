import os
import tempfile
import base64

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.path.join(
    "src",
    "runs",
    "detect",
    "yolov8s_all_countries_custom2",
    "weights",
    "best.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)

    # Use CPU on Streamlit Cloud
    model.to(DEVICE)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model()

except Exception as e:
    st.error("Failed to load YOLO model.")
    st.exception(e)
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_detections(frame, threshold=0.5):
    """
    Run YOLO inference and return detection data.
    """

    results = model(
        frame,
        verbose=False,
        conf=threshold
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = [
                float(coord)
                for coord in box.xyxy[0].cpu().numpy()
            ]

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": class_name,
                "confidence": confidence
            })

    return detections


def process_frame(frame, threshold=0.5):
    """
    Run YOLO inference and draw bounding boxes.
    """

    # Make a copy so the original frame is not modified
    output = frame.copy()

    detections = get_detections(
        output,
        threshold
    )

    for detection in detections:

        x1 = int(detection["x1"])
        y1 = int(detection["y1"])
        x2 = int(detection["x2"])
        y2 = int(detection["y2"])

        class_name = detection["class"]
        confidence = detection["confidence"]

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output, detections


def process_image(uploaded_file, threshold):
    """
    Process an uploaded image.
    """

    file_bytes = np.frombuffer(
        uploaded_file.read(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError("Could not decode image.")

    processed_frame, detections = process_frame(
        frame,
        threshold
    )

    return processed_frame, detections


def process_video(uploaded_file, threshold):
    """
    Process an uploaded video.

    Returns:
        processed video path
        detection summary
    """

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    # Temporary input file
    input_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    input_temp.write(
        uploaded_file.read()
    )

    input_temp.close()

    input_path = input_temp.name

    # Open video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        os.remove(input_path)
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    # Temporary output video
    output_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_temp.close()

    output_path = output_temp.name

    # MP4 codec
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    results_data = []

    frame_id = 0

    progress_bar = st.progress(0)

    status = st.empty()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frame, detections = process_frame(
            frame,
            threshold
        )

        writer.write(
            processed_frame
        )

        if detections:

            results_data.append({
                "timestamp": frame_id / fps,
                "pothole_count": len(detections),
                "detections": detections
            })

        frame_id += 1

        if frame_count > 0:

            progress = min(
                frame_id / frame_count,
                1.0
            )

            progress_bar.progress(
                progress
            )

            status.write(
                f"Processing frame "
                f"{frame_id}/{frame_count}"
            )

    cap.release()
    writer.release()

    progress_bar.progress(1.0)

    status.write(
        f"Completed: {frame_id} frames"
    )

    os.remove(input_path)

    return (
        output_path,
        results_data,
        fps,
        duration
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

st.sidebar.info(
    f"Running on: {DEVICE}"
)

st.sidebar.markdown(
    """
    ### Supported files

    **Images**
    - JPG
    - JPEG
    - PNG

    **Videos**
    - MP4
    - AVI
    - MOV
    - MKV
    """
)


# ============================================================
# HEADER
# ============================================================

st.title("🚧 Road Damage Detection")

st.write(
    "Upload a road image or video and detect "
    "road damage using your trained YOLO model."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    filename = uploaded_file.name.lower()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if filename.endswith(
        (".jpg", ".jpeg", ".png")
    ):

        st.subheader("🖼️ Image Detection")

        col1, col2 = st.columns(2)

        with col1:

            st.write("Original")

            uploaded_file.seek(0)

            original_bytes = uploaded_file.read()

            st.image(
                original_bytes,
                use_container_width=True
            )

        if st.button(
            "🔍 Detect Road Damage",
            type="primary"
        ):

            with st.spinner(
                "Running YOLO detection..."
            ):

                uploaded_file.seek(0)

                processed_frame, detections = process_image(
                    uploaded_file,
                    threshold
                )

            with col2:

                st.write(
                    "Detection Result"
                )

                # OpenCV BGR -> RGB
                processed_rgb = cv2.cvtColor(
                    processed_frame,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    processed_rgb,
                    use_container_width=True
                )

            st.success(
                f"Detected {len(detections)} "
                f"road damage object(s)."
            )

            # ------------------------------------------------
            # DETECTION TABLE
            # ------------------------------------------------

            if detections:

                st.subheader(
                    "📊 Detections"
                )

                table_data = []

                for i, detection in enumerate(
                    detections,
                    start=1
                ):

                    table_data.append({
                        "ID": i,
                        "Class": detection["class"],
                        "Confidence": round(
                            detection["confidence"],
                            3
                        ),
                        "X1": round(
                            detection["x1"],
                            1
                        ),
                        "Y1": round(
                            detection["y1"],
                            1
                        ),
                        "X2": round(
                            detection["x2"],
                            1
                        ),
                        "Y2": round(
                            detection["y2"],
                            1
                        )
                    })

                st.dataframe(
                    table_data,
                    use_container_width=True
                )


    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    elif filename.endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):

        st.subheader("🎥 Video Detection")

        st.video(
            uploaded_file
        )

        if st.button(
            "🔍 Process Video",
            type="primary"
        ):

            with st.spinner(
                "Processing video..."
            ):

                uploaded_file.seek(0)

                (
                    output_path,
                    results_data,
                    fps,
                    duration
                ) = process_video(
                    uploaded_file,
                    threshold
                )

            st.success(
                "Video processing completed!"
            )

            # ------------------------------------------------
            # RESULT VIDEO
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            st.subheader(
                "🎬 Processed Video"
            )

            st.video(
                video_bytes
            )

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name=(
                    "road_damage_detected.mp4"
                ),
                mime="video/mp4"
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Detection Summary"
            )

            total_detection_frames = len(
                results_data
            )

            total_detections = sum(
                len(item["detections"])
                for item in results_data
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Video Duration",
                    f"{duration:.2f}s"
                )

            with col2:

                st.metric(
                    "Frames With Detection",
                    total_detection_frames
                )

            with col3:

                st.metric(
                    "Total Detections",
                    total_detections
                )

            # ------------------------------------------------
            # DETECTION DETAILS
            # ------------------------------------------------

            if results_data:

                st.subheader(
                    "🔎 Detection Details"
                )

                for item in results_data:

                    timestamp = item[
                        "timestamp"
                    ]

                    detections = item[
                        "detections"
                    ]

                    st.write(
                        f"**Time: "
                        f"{timestamp:.2f}s** — "
                        f"{len(detections)} detection(s)"
                    )

                    for detection in detections:

                        st.write(
                            f"- "
                            f"{detection['class']} "
                            f"({detection['confidence']:.2%})"
                        )

            # Clean output
            try:
                os.remove(output_path)
            except Exception:
                pass


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Road Damage Detection • YOLO • OpenCV • Streamlit"
)
