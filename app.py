from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from flask import session
from PIL import Image
from io import BytesIO
import base64
import sqlite3
import os
import math
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "WasteVision AI_secret_key"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "biowaste_best_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "history.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

# Keep this order exactly as your training class order
CLASS_NAMES = {
    0: "Unused Tablets",
    1: "Unused Syringe",
    2: "Waste Syringe",
    3: "Waste Tablets"
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Prediction history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            source TEXT,
            created_at TEXT
        )
    """)

    # User table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # Feedback Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            rating TEXT,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
init_db()
def preprocess_pil_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr

def predict_array(arr):
    preds = model.predict(arr, verbose=0)[0]
    class_index = np.argmax(preds)
    confidence = float(preds[class_index]) * 100
    predicted_class = CLASS_NAMES[class_index]
    print("Predictions:", preds)
    print("Predicted:", predicted_class)
    return predicted_class, round(confidence, 2)

def save_history(filename, predicted_class, confidence, source):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (filename, predicted_class, confidence, source, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (filename, predicted_class, confidence, source, __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return redirect(url_for("index"))

@app.route("/live-detection")
def live_detection():

    if "user_id" not in session:
        return redirect(url_for("login_signup"))

    return render_template("live_detection.html")

@app.route("/upload-image")
def upload_image():
    return render_template("upload_image.html")

# @app.route("/detection-history")
# def detection_history():
#
#     if "user_id" not in session:
#         return redirect(url_for("login_signup"))
#
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute("""
#     SELECT filename,
#            predicted_class,
#            confidence,
#            source,
#            created_at
#     FROM predictions
#     ORDER BY id DESC
#     """)
#     rows = cur.fetchall()
#     conn.close()
#     return render_template("detection_history.html", rows=rows)
@app.route("/detection-history")
def detection_history():
    if "user_id" not in session:
        return redirect(url_for("login_signup"))
    page = request.args.get("page", 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, (per_page, offset))

    rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]

    conn.close()

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "detection_history.html",
        rows=rows,
        page=page,
        total_pages=total_pages
    )
@app.route("/waste-guide")
def waste_guide():
    return render_template("waste_guide.html")

@app.route("/guide")
def guide():
    return redirect(url_for("waste_guide"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/login-signup")
def login_signup():
    return render_template("login_signup.html")


@app.route("/signup", methods=["POST"])
def signup():

    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users(username,email,password) VALUES(?,?,?)",
            (username, email, password)
        )

        conn.commit()

        return redirect(url_for("login_signup"))

    except:
        return "User already exists"

    finally:
        conn.close()


@app.route("/login", methods=["POST"])
def login():

    email = request.form["email"]
    password = request.form["password"]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, password)
    )

    user = cur.fetchone()

    conn.close()

    if user:
        print("LOGIN SUCCESS:", user)

        session["user_id"] = user[0]
        session["username"] = user[1]

        return redirect(url_for("index"))

    return "Invalid Email or Password"


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    img = Image.open(save_path)
    arr = preprocess_pil_image(img)

    predicted_class, confidence = predict_array(arr)
    save_history(filename, predicted_class, confidence, "upload")

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence
    })

@app.route("/predict-frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image data"}), 400

    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes))
        arr = preprocess_pil_image(img)
        predicted_class, confidence = predict_array(arr)

        if confidence > 75:
            save_history(
                "live_camera",
                predicted_class,
                confidence,
                "live"
            )

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():

    name = request.form["name"]
    email = request.form["email"]
    rating = request.form["rating"]
    feedback = request.form["feedback"]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO feedback(name,email,rating,feedback)
        VALUES(?,?,?,?)
    """, (name,email,rating,feedback))

    conn.commit()
    conn.close()

    return redirect(url_for("contact"))
@app.route("/history/<int:id>")
def history_detail(id):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM predictions
        WHERE id = ?
    """, (id,))

    row = cur.fetchone()

    conn.close()

    if row is None:
        return "History record not found"

    return render_template(
        "history_detail.html",
        row=row
    )
@app.route("/delete-history/<int:id>")
def delete_history(id):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM predictions WHERE id=?",
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for("detection_history"))
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
