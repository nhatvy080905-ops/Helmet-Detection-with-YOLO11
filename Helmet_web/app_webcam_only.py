import os

import cv2
from flask import Flask, Response, render_template, request, url_for

from app_webcam import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    create_status_frame,
    draw_webcam_detection,
    encode_stream_frame,
    generate_webcam_frames,
)


app = Flask(__name__)


def parse_camera_source(source):
    source = str(source).strip()

    if source.isdigit():
        return int(source)

    return source


def get_request_config():
    source = request.args.get("source", os.environ.get("CAMERA_SOURCE", "0")).strip()
    width = request.args.get("width", os.environ.get("CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH))
    height = request.args.get("height", os.environ.get("CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT))

    try:
        width = int(width)
    except ValueError:
        width = DEFAULT_CAMERA_WIDTH

    try:
        height = int(height)
    except ValueError:
        height = DEFAULT_CAMERA_HEIGHT

    return source, width, height


def open_camera(camera_source):
    if isinstance(camera_source, int):
        return cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)

    return cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)


def generate_frames(source, width, height):
    yield from generate_webcam_frames(source, width, height)


@app.route("/")
def index():
    source, width, height = get_request_config()
    start = request.args.get("start") == "1"
    stream_url = None

    if start:
        stream_url = url_for(
            "video_feed",
            source=source,
            width=width,
            height=height
        )

    return render_template(
        "webcam.html",
        source=source,
        width=width,
        height=height,
        start=start,
        stream_url=stream_url
    )


@app.route("/video_feed")
def video_feed():
    source, width, height = get_request_config()

    return Response(
        generate_frames(source, width, height),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, threaded=True, use_reloader=False, port=5001)
