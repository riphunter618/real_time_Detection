import base64
import cv2
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ✅ Change this to your remote detector (ngrok or Render backend)
DETECTOR_URL = "https://hypocycloidal-felicidad-uncontributively.ngrok-free.dev/detect"

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def draw_boxes(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det.get("class", "Object")
        name = det.get("name", "Unknown")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} - {name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


@app.post("/process_frame")
async def process_frame(request: Request):
    """
    Receive a frame from the frontend, send it to the detection backend,
    draw boxes, and return detection results (or processed image).
    """
    data = await request.json()
    image_base64 = data.get("image")

    if not image_base64:
        return {"error": "No image data received"}

    try:
        # Decode Base64 image
        image_bytes = base64.b64decode(image_base64.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to JPEG to send to detection backend
        _, buffer = cv2.imencode(".jpg", frame)
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

        # Send to remote detector
        response = requests.post(DETECTOR_URL, files=files, timeout=10)
        if response.status_code == 200:
            results = response.json().get("detections", [])
        else:
            results = []

        # Draw boxes locally
        #print(results[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_with_boxes = draw_boxes(frame, results)
        logs = response.json().get("message", [])
        # Convert processed frame to Base64 to send back
        _, jpeg = cv2.imencode(".jpg", frame_with_boxes)
        # cv2.imshow('boom', jpeg)
        processed_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

        return {
            "message": "Frame processed successfully",
            "detections": results,
            "processed_image": f"data:image/jpeg;base64,{processed_b64}",
            "logs":logs
        }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    # Use the PORT environment variable Render provides, fallback to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("split1:app", host="0.0.0.0", port=port, reload=True)
