import os
import sys
import shutil
import subprocess
import datetime
import pickle
from collections import Counter

import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth

# ─────────────────────────────────────────────
# Anchor all paths to this file's location
# (fixes ModuleNotFoundError in uvicorn subprocesses)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Anti-spoofing imports ──
ANTI_SPOOF_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "Silent-Face-Anti-Spoofing"))
sys.path.insert(0, ANTI_SPOOF_ROOT)
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# ─────────────────────────────────────────────
# App & Middleware
# ─────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Firebase
# ─────────────────────────────────────────────
cred = credentials.Certificate(os.path.join(BASE_DIR, "face-log-fb54d-firebase-adminsdk-fbsvc-e64cc3dab5.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# ─────────────────────────────────────────────
# Models  (absolute paths — safe in subprocesses)
# ─────────────────────────────────────────────
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "model"))

yolo_model = YOLO(os.path.join(MODEL_DIR, "best.pt"))

with open(os.path.join(MODEL_DIR, "face_recognition_knn.pkl"), "rb") as f:
    knn_clf, label_encoder = pickle.load(f)


# ─────────────────────────────────────────────
# Model Hot-Reload Helper
# ─────────────────────────────────────────────
def reload_model():
    """Re-read the KNN model from disk into the global variables."""
    global knn_clf, label_encoder
    with open(os.path.join(MODEL_DIR, "face_recognition_knn.pkl"), "rb") as f:
        knn_clf, label_encoder = pickle.load(f)


# ── Anti-spoofing setup ──
DEVICE_ID = 0
SPOOF_MODEL_DIR = os.path.join(ANTI_SPOOF_ROOT, "resources", "anti_spoof_models")

# Temporarily switch to the repo directory so internal relative paths resolve
_original_dir = os.getcwd()
os.chdir(ANTI_SPOOF_ROOT)
anti_spoof = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()
os.chdir(_original_dir)


# ─────────────────────────────────────────────
# Anti-Spoofing Helper
# ─────────────────────────────────────────────
def is_real_face(rgb_img: np.ndarray, box_css: tuple) -> bool:
    top, right, bottom, left = box_css
    bbox = [left, top, right - left, bottom - top]

    prediction = np.zeros((1, 3))
    for model_name in os.listdir(SPOOF_MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": rgb_img,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        img_patch = image_cropper.crop(**param)
        prediction += anti_spoof.predict(
            img_patch, os.path.join(SPOOF_MODEL_DIR, model_name)
        )

    label = np.argmax(prediction)
    return int(label) == 1


# ─────────────────────────────────────────────
# KNN Recognition Helper
# FIX 1: distance checked BEFORE predict()
# FIX 2: majority vote with N neighbours
# FIX 3: tighter threshold (0.4 instead of 0.5)
# ─────────────────────────────────────────────
DISTANCE_THRESHOLD = 0.4

def knn_recognize(encoding):
    """
    Returns (name, uid) if the encoding matches a known student,
    or (None, None) if it is Unknown.
    """
    n_samples = len(knn_clf._fit_X)
    N = min(5, n_samples)

    distances, indices = knn_clf.kneighbors([encoding], n_neighbors=N)

    # Reject immediately if the single nearest neighbour is too far
    if distances[0][0] > DISTANCE_THRESHOLD:
        return None, None

    # Collect only neighbours within threshold
    close_indices = [
        idx for idx, dist in zip(indices[0], distances[0])
        if dist <= DISTANCE_THRESHOLD
    ]

    if not close_indices:
        return None, None

    # Majority vote among close neighbours
    labels = [label_encoder.inverse_transform([knn_clf._y[i]])[0] for i in close_indices]
    winner, count = Counter(labels).most_common(1)[0]

    # Require strict majority agreement
    if count < (len(close_indices) // 2 + 1):
        return None, None

    parts = winner.split(" ")
    uid = parts[-1]
    name = " ".join(parts[:-1])
    return name, uid


# ─────────────────────────────────────────────
# DUPLICATE FACE CHECK (used during enrollment)
# ─────────────────────────────────────────────
@app.post("/check-face")
async def check_face(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = yolo_model(rgb, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((y1, x2, y2, x1))

    if not boxes:
        return {"status": "no_face"}

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        name, uid = knn_recognize(encoding)
        if name:
            return {"status": "duplicate", "name": name, "uid": uid}

    return {"status": "unknown"}


# ─────────────────────────────────────────────
# ATTENDANCE
# FIX: processes ALL faces, returns list
# FIX: uses knn_recognize (threshold before predict + majority vote)
# ─────────────────────────────────────────────
@app.post("/recognize")
async def recognize(file: UploadFile = File(...), class_id: str = Form("")):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results_yolo = yolo_model(rgb, verbose=False)
    boxes = []
    for result in results_yolo:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((y1, x2, y2, x1))

    if not boxes:
        return {"results": [], "message": "No Face"}

    encodings = face_recognition.face_encodings(rgb, boxes)
    attended = []

    for box, encoding in zip(boxes, encodings):

        # Anti-spoofing check — flag but continue for other faces
        if not is_real_face(rgb, box):
            attended.append({"name": "Spoof", "uid": None, "time": None})
            continue

        name, uid = knn_recognize(encoding)

        if not name:
            attended.append({"name": "Unknown", "uid": None, "time": None})
            continue

        timestamp = datetime.datetime.now()

        # Mark attendance in Firestore
        if class_id:
            class_ref = db.collection("classes").document(class_id)
            class_ref.update({"attended": firestore.ArrayUnion([uid])})

        attended.append({
            "name": name,
            "uid": uid,
            "time": timestamp.strftime("%H:%M:%S"),
        })

    return {"results": attended}


# ─────────────────────────────────────────────
# SUBJECTS & CLASSES
# ─────────────────────────────────────────────
@app.get("/subjects")
def get_subjects():
    subjects_ref = db.collection("subjects").stream()
    return [
        {"id": doc.id, "name": doc.to_dict().get("name", doc.id)}
        for doc in subjects_ref
    ]


@app.get("/classes/{subject_id}")
def get_classes(subject_id: str):
    classes_ref = db.collection("classes").where("subject", "==", subject_id).stream()
    now = datetime.datetime.now(datetime.timezone.utc)
    classes = []
    for doc in classes_ref:
        data = doc.to_dict()
        date = data.get("date")
        duration = data.get("duration", 0)

        if not date:
            continue
        if date.tzinfo is None:
            date = date.replace(tzinfo=datetime.timezone.utc)

        class_end = date + datetime.timedelta(minutes=int(duration))
        if not (date <= now <= class_end):
            continue

        date_str = date.strftime("%Y-%m-%d %H:%M")
        classes.append({
            "id": doc.id,
            "display": f"{date_str} | {data.get('location', 'Unknown')}",
        })
    return classes


# ─────────────────────────────────────────────
# ENROLLMENT
# ─────────────────────────────────────────────
@app.get("/departments")
def get_departments():
    docs = db.collection("departments").stream()
    return [
        {
            "id": doc.id,
            "code": doc.to_dict().get("code"),
            "name": doc.to_dict().get("name", doc.id),
        }
        for doc in docs
    ]


@app.get("/groups/{department_id}")
def get_groups(department_id: str):
    docs = list(
        db.collection("groups").where("departmentId", "==", department_id).stream()
    )
    if not docs:
        dept_doc = db.collection("departments").document(department_id).get()
        if dept_doc.exists:
            dept_code = dept_doc.to_dict().get("code")
            if dept_code:
                docs = list(
                    db.collection("groups").where("departmentId", "==", dept_code).stream()
                )
    return [{"id": doc.id, "name": doc.to_dict().get("name", doc.id)} for doc in docs]


@app.post("/enroll")
async def enroll_student(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    department: str = Form(...),
    group: str = Form(...),
    photos: List[UploadFile] = File(...),
):
    # 1. Check duplicate email
    try:
        fb_auth.get_user_by_email(email)
        raise HTTPException(status_code=400, detail="Email already registered.")
    except firebase_admin.auth.UserNotFoundError:
        pass

    # 2. Create Firebase Auth user
    user = fb_auth.create_user(email=email, password=password, display_name=name)
    uid = user.uid

    # 3. Save student document
    db.collection("students").document(uid).set({
        "uid": uid,
        "name": name,
        "email": email,
        "department": department,
        "group": group,
    })

    # 4. Add student to group
    db.collection("groups").document(group).update(
        {"students": firestore.ArrayUnion([uid])}
    )

    # 5. Save images to dataset
    formatted_name = name.strip().replace(" ", "_")
    dataset_path = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))
    face_dir = os.path.join(dataset_path, f"{formatted_name} {uid}")
    os.makedirs(face_dir, exist_ok=True)
    print("Saving images to:", face_dir)

    saved_paths = []
    for i, photo in enumerate(photos):
        contents = await photo.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        path = os.path.join(face_dir, f"{i:03d}.jpg")
        cv2.imwrite(path, img)
        saved_paths.append(path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid images uploaded.")

    # 6. Anti-spoofing check on saved images
    real_count = 0
    for path in saved_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        rgb_enroll = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_boxes = face_recognition.face_locations(rgb_enroll)
        if len(face_boxes) == 1 and is_real_face(rgb_enroll, face_boxes[0]):
            real_count += 1

    if real_count < len(saved_paths) // 2:
        shutil.rmtree(face_dir, ignore_errors=True)
        fb_auth.delete_user(uid)
        db.collection("students").document(uid).delete()
        raise HTTPException(
            status_code=400,
            detail=f"Liveness failed ({real_count}/{len(saved_paths)}). Use real face, good lighting.",
        )

    # 7. Run embedding script
    try:
        subprocess.run(
            [sys.executable, "new_student_embed.py", formatted_name, uid],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Embedding failed:", e)
        raise HTTPException(status_code=500, detail="Embedding process failed. Check dataset path.")

    # 8. Reload model
    reload_model()
    return {"uid": uid, "name": name}


# ─────────────────────────────────────────────
# STUDENT LIST
# ─────────────────────────────────────────────
@app.get("/students")
def get_students():
    docs = db.collection("students").stream()
    return [
        {
            "uid":        doc.id,
            "name":       doc.to_dict().get("name", ""),
            "email":      doc.to_dict().get("email", ""),
            "department": doc.to_dict().get("department", ""),
            "group":      doc.to_dict().get("group", ""),
        }
        for doc in docs
    ]


# ─────────────────────────────────────────────
# DELETE STUDENT
# ─────────────────────────────────────────────
@app.delete("/students/{uid}")
async def delete_student(uid: str):
    student_ref = db.collection("students").document(uid)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found.")

    data = student_doc.to_dict()
    name = data.get("name", "")
    group_id = data.get("group", "")
    formatted_name = name.strip().replace(" ", "_")

    try:
        fb_auth.delete_user(uid)
    except firebase_admin.auth.UserNotFoundError:
        pass

    student_ref.delete()

    if group_id:
        try:
            db.collection("groups").document(group_id).update(
                {"students": firestore.ArrayRemove([uid])}
            )
        except Exception as e:
            print(f"Warning: could not update group {group_id}: {e}")

    try:
        subprocess.run(
            [sys.executable, "delete_student.py", formatted_name, uid],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Delete/retrain script failed:", e)
        raise HTTPException(
            status_code=500,
            detail="Firestore records removed but model retraining failed. Check logs.",
        )

    reload_model()
    return {"detail": f"Student '{name}' (UID: {uid}) deleted and model retrained."}