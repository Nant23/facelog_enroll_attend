import os
import sys
import shutil
import subprocess
import datetime
import pickle

import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth

# ── Anti-spoofing imports ──
sys.path.insert(0, "./Silent-Face-Anti-Spoofing")
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
cred = credentials.Certificate("face-log-fb54d-firebase-adminsdk-fbsvc-e64cc3dab5.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
yolo_model = YOLO("model/best.pt")

with open("model/face_recognition_knn.pkl", "rb") as f:
    knn_clf, label_encoder = pickle.load(f)


# ─────────────────────────────────────────────
# Model Hot-Reload Helper
# ─────────────────────────────────────────────
def reload_model():
    """Re-read the KNN model from disk into the global variables."""
    global knn_clf, label_encoder
    with open("model/face_recognition_knn.pkl", "rb") as f:
        knn_clf, label_encoder = pickle.load(f)

# ── Anti-spoofing setup ──
DEVICE_ID = 0
ANTI_SPOOF_ROOT = os.path.abspath("./Silent-Face-Anti-Spoofing")
SPOOF_MODEL_DIR = os.path.join(ANTI_SPOOF_ROOT, "resources", "anti_spoof_models")

# Temporarily switch to the repo directory so internal relative paths resolve
_original_dir = os.getcwd()
os.chdir(ANTI_SPOOF_ROOT)
anti_spoof = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()
os.chdir(_original_dir)  # Switch back immediately


# ─────────────────────────────────────────────
# Anti-Spoofing Helper
# ─────────────────────────────────────────────
def is_real_face(rgb_img: np.ndarray, box_css: tuple) -> bool:
    """
    Runs the Silent-Face liveness check on a single detected face.

    Args:
        rgb_img:  Full frame in RGB (H x W x 3).
        box_css:  Face bounding box as (top, right, bottom, left)
                  — the same format used by face_recognition / YOLO output.

    Returns:
        True  → real / live face
        False → spoof (photo, screen, mask …)
    """
    top, right, bottom, left = box_css
    # CropImage expects (x, y, w, h)
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

    # label 1 → real, label 0 → spoof
    label = np.argmax(prediction)
    return int(label) == 1


# ─────────────────────────────────────────────
# DUPLICATE FACE CHECK (used during enrollment)
# ─────────────────────────────────────────────

@app.post("/check-face")
async def check_face(file: UploadFile = File(...)):
    """
    Lightweight endpoint called during enrollment capture to detect
    whether the face in the frame already belongs to an enrolled student.

    Returns:
        { "status": "unknown" }           — no face or unrecognised face
        { "status": "no_face" }           — no face detected at all
        { "status": "duplicate",
          "name": "<student name>",
          "uid":  "<student uid>" }       — already enrolled student detected
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Face detection via YOLO ──
    results = yolo_model(rgb, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((y1, x2, y2, x1))  # → (top, right, bottom, left)

    if not boxes:
        return {"status": "no_face"}

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        distances, _ = knn_clf.kneighbors([encoding], n_neighbors=1)

        # Distance threshold — same as /recognize
        if distances[0][0] > 0.5:
            continue  # unrecognised face, keep checking others

        predicted = knn_clf.predict([encoding])[0]
        label = label_encoder.inverse_transform([predicted])[0]

        parts = label.split(" ")
        uid  = parts[-1]
        name = " ".join(parts[:-1])

        return {"status": "duplicate", "name": name, "uid": uid}

    return {"status": "unknown"}


# ─────────────────────────────────────────────
# ATTENDANCE
# ─────────────────────────────────────────────

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), class_id: str = Form("")):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Face detection via YOLO ──
    results = yolo_model(rgb, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((y1, x2, y2, x1))  # → (top, right, bottom, left)

    if not boxes:
        return {"name": "No Face"}

    encodings = face_recognition.face_encodings(rgb, boxes)

    for box, encoding in zip(boxes, encodings):

        # ── Anti-spoofing check ──
        if not is_real_face(rgb, box):
            return {"name": "Spoof", "uid": None, "time": None}

        # ── KNN recognition ──
        distances, _ = knn_clf.kneighbors([encoding], n_neighbors=1)
        predicted = knn_clf.predict([encoding])[0]
        label = label_encoder.inverse_transform([predicted])[0]

        if distances[0][0] > 0.5:
            return {"name": "Unknown"}

        timestamp = datetime.datetime.now()
        parts = label.split(" ")
        uid = parts[-1]
        name = " ".join(parts[:-1])

        # ── Mark attendance in Firestore ──
        if class_id:
            class_ref = db.collection("classes").document(class_id)
            class_ref.update({"attended": firestore.ArrayUnion([uid])})

        return {"name": name, "uid": uid, "time": timestamp.strftime("%H:%M:%S")}

    return {"name": "No Face"}


@app.get("/subjects")
def get_subjects():
    subjects_ref = db.collection("subjects").stream()
    return [
        {"id": doc.id, "name": doc.to_dict().get("name", doc.id)}
        for doc in subjects_ref
    ]


@app.get("/classes/{subject_id}")
def get_classes(subject_id: str):
    classes_ref = (
        db.collection("classes").where("subject", "==", subject_id).stream()
    )
    classes = []
    for doc in classes_ref:
        data = doc.to_dict()
        date = data.get("date")
        date_str = date.strftime("%Y-%m-%d %H:%M") if date else "No Date"
        classes.append(
            {
                "id": doc.id,
                "display": f"{date_str} | {data.get('location', 'Unknown')}",
            }
        )
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
    # Primary: match by Firestore document ID (the canonical approach)
    docs = list(
        db.collection("groups").where("departmentId", "==", department_id).stream()
    )

    # Fallback: if nothing found, try matching by department code field.
    # This handles databases where groups were enrolled using the dept code
    # rather than the Firestore doc ID as the departmentId value.
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
    # ── 1. Check duplicate email ──
    try:
        fb_auth.get_user_by_email(email)
        raise HTTPException(status_code=400, detail="Email already registered.")
    except firebase_admin.auth.UserNotFoundError:
        pass

    # ── 2. Create Firebase Auth user ──
    user = fb_auth.create_user(email=email, password=password, display_name=name)
    uid = user.uid

    # ── 3. Save student document ──
    db.collection("students").document(uid).set(
        {
            "uid": uid,
            "name": name,
            "email": email,
            "department": department,
            "group": group,
        }
    )

    # ── 4. Add student to group ──
    db.collection("groups").document(group).update(
        {"students": firestore.ArrayUnion([uid])}
    )

    # ── 5. SAVE IMAGES → ../dataset ──
    formatted_name = name.strip().replace(" ", "_")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "..", "dataset")

    # IMPORTANT: match embed script format → "name uid"
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

    if len(saved_paths) == 0:
        raise HTTPException(status_code=400, detail="No valid images uploaded.")

    # ── 6. Anti-spoofing check ──
    real_count = 0

    for path in saved_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        rgb_enroll = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_boxes = face_recognition.face_locations(rgb_enroll)

        if len(face_boxes) == 1 and is_real_face(rgb_enroll, face_boxes[0]):
            real_count += 1

    MIN_REAL_FRAMES = len(saved_paths) // 2

    if real_count < MIN_REAL_FRAMES:
        # rollback everything
        shutil.rmtree(face_dir, ignore_errors=True)
        fb_auth.delete_user(uid)
        db.collection("students").document(uid).delete()

        raise HTTPException(
            status_code=400,
            detail=(
                f"Liveness failed ({real_count}/{len(saved_paths)}). "
                "Use real face, good lighting."
            ),
        )

    # ── 7. Run embedding script ──
    try:
        subprocess.run(
            [sys.executable, "new_student_embed.py", formatted_name, uid],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Embedding failed:", e)

        raise HTTPException(
            status_code=500,
            detail="Embedding process failed. Check dataset path.",
        )

    # ── 8. Reload model into memory ──
    reload_model()

    return {"uid": uid, "name": name}


# ─────────────────────────────────────────────
# STUDENT LIST  (for the delete UI)
# ─────────────────────────────────────────────

@app.get("/students")
def get_students():
    """Return all enrolled students from Firestore."""
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
    """
    Fully remove a student:
      1. Firebase Auth user
      2. Firestore student document
      3. Remove UID from their group's students array
      4. Delete dataset folder
      5. Retrain KNN model without their embeddings
      6. Hot-reload the model in memory
    """
    # ── 1. Fetch student record ──
    student_ref = db.collection("students").document(uid)
    student_doc = student_ref.get()

    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found.")

    data       = student_doc.to_dict()
    name       = data.get("name", "")
    group_id   = data.get("group", "")
    formatted_name = name.strip().replace(" ", "_")

    # ── 2. Delete Firebase Auth user ──
    try:
        fb_auth.delete_user(uid)
    except firebase_admin.auth.UserNotFoundError:
        pass  # already gone — keep going

    # ── 3. Delete Firestore student document ──
    student_ref.delete()

    # ── 4. Remove UID from group array ──
    if group_id:
        try:
            db.collection("groups").document(group_id).update(
                {"students": firestore.ArrayRemove([uid])}
            )
        except Exception as e:
            print(f"Warning: could not update group {group_id}: {e}")

    # ── 5. Run delete + retrain script ──
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

    # ── 6. Hot-reload model ──
    reload_model()

    return {"detail": f"Student '{name}' (UID: {uid}) deleted and model retrained."}
