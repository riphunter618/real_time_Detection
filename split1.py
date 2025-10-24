# fastapi_service.py
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from io import BytesIO
from PIL import Image
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# URL of your remote detection backend (Colab)
DETECTOR_URL = "https://hypocycloidal-felicidad-uncontributively.ngrok-free.dev/detect"


def draw_boxes(frame, detections):
    """Draw boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det.get("class", "Object")
        name = det.get("name", "Unknown")

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label
        cv2.putText(frame, f"{cls_name} - {name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:
        # Encode frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

            try:
                # Send frame to detection backend
                response = requests.post(DETECTOR_URL, files=files, timeout=5)
                results = response.json().get("detections", [])
            except:
                results = []

            # Draw boxes locally
            frame = draw_boxes(frame, results)

            # Encode frame for streaming
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

    cap.release()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Use the PORT environment variable Render provides, fallback to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("split1:app", host="0.0.0.0", port=port, reload=True)


