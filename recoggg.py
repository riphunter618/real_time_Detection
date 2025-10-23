from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings

warnings.filterwarnings('ignore')
from deepface import DeepFace
import cv2, numpy as np, torch, warnings, time
from datetime import datetime
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

warnings.filterwarnings("ignore")
DB_CONFIG = {
    "host": "localhost",
    "database": "recog",
    "user": "postgres",
    "password": "cobbvanth618"
}
pool = SimpleConnectionPool(1, 50, **DB_CONFIG)


def get_conn():
    return pool.getconn()


def put_conn(conn):
    pool.putconn(conn)


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH).to(device)
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
known_faces = {}
recognition_enabled = True

# 🧾 In-memory logs list
logs = []
table_name = 'faces1'


recently_logged = set()

def add_log(message: str):
    ts = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{ts}] {message}"

    # avoid duplicates
    if message not in recently_logged:
        logs.append(log_entry)
        recently_logged.add(message)
        print(log_entry)

        # keep memory small
        if len(recently_logged) > 50:
            recently_logged.clear()

    if len(logs) > 100:
        logs.pop(0)


def cosine_distance(emb1, emb2):
    emb1, emb2 = np.array(emb1), np.array(emb2)
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def verify(query_str):
    conn = get_conn()  # verifying the image with the db
    # logging.info('connected to db')
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        f"""
        SELECT name,image_url , embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY embedding <-> %s::vector
        LIMIT 5
        """,
        (query_str, query_str))
    # logging.info('verification started')
    x = cursor.fetchall()
    cursor.close()
    put_conn(conn)
    # logging.info('connection returned to pool')
    return x


def generate_frames():
    frame_count = 0
    current_objects = set()
    global recognition_enabled
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            results = model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                cls_name = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_name])

            tracks = tracker.update_tracks(detections, frame=frame)
            new_objects = set()
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cls_name = track.get_det_class() if hasattr(track, 'get_det_class') else "Object"
                new_objects.add(cls_name)
                face_crop = frame[y1:y2, x1:x2]
                name = "Detecting..."

                if recognition_enabled and cls_name.lower() == "person":
                    if track_id not in known_faces:
                        try:
                            rep = DeepFace.represent(face_crop, model_name='ArcFace', enforce_detection=False)
                            embedding = rep[0]["embedding"]
                            if rep:
                                #embedding = rep[0]["embedding"]
                                query_str = "[" + ",".join(str(x) for x in embedding) + "]"
                                db_result = verify(query_str)
                                min_dist, identity = 1.0, "Unknown"
                                if db_result and db_result[0]["distance"] < 4.5:
                                    identity = db_result[0]["name"]
                                    known_faces[track_id] = {"name": identity, "embedding": embedding}
                                    add_log(f"✅ Recognized: {identity}")
                            else:
                                name = f"Unknown_{len(known_faces) + 1}"
                                known_faces[track_id] = {"name": name, "embedding": embedding}
                                add_log(f"🆕 New face detected: {name}")
                        except Exception as e:
                            add_log(f"Recognition error: {e}")

                name = known_faces.get(track_id, {}).get("name", "Detecting...")
                #print(known_faces)
                conf_text = f"{cls_name} (ID {track_id}) - {name}"
                add_log(f"Detected {cls_name} [{track_id}]")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, conf_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for obj in new_objects - current_objects:
                add_log(f"🔍 New object detected: {obj}")
                # Update current objects
            current_objects = new_objects

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            # add_log(f"Detected {cls_name} [{track_id}]")
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/toggle_recognition")
async def toggle_recognition():
    global recognition_enabled
    recognition_enabled = not recognition_enabled
    add_log(f"🔁 Face Recognition toggled: {'ON' if recognition_enabled else 'OFF'}")
    return {"recognition_enabled": recognition_enabled}


@app.get("/logs")
async def get_logs():
    """Return the last 50 log messages"""
    return JSONResponse({"logs": logs[-50:]})
