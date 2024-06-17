import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response, g
from datetime import datetime
from PIL import Image, ImageDraw
import os
import time
import torch
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import requests
import io
import cv2
import tensorflow as tf
import pytesseract
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse
from zipfile import ZipFile
import multiprocessing
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from ultralytics import YOLO
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
import re
import sqlite3
import mysql.connector
import argparse
from flask import Flask, render_template, request
import mysql.connector
from mysql.connector import Error
import os
from flask_sqlalchemy import SQLAlchemy
import argparse
from flask import Flask, render_template, request, redirect, url_for
import base64

app = Flask(__name__)


#user Authentication (Login, Logout, Check Login)
################################################################################################
app.secret_key = 'key'  # Change this to a random secret key

# MySQL database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'carplateidentification'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/carplateidentification'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class DetectedCarDetails(db.Model):
    __tablename__ = 'detectedcardetails'
    captured_details_id = db.Column(db.Integer, primary_key=True)
    detected_car_model = db.Column(db.String(50))
    detected_car_picture = db.Column(db.LargeBinary)
    detected_color = db.Column(db.String(20))
    detected_plate_number = db.Column(db.String(10))
    detected_plate_picture = db.Column(db.LargeBinary)
    detected_plate_origin = db.Column(db.String(20))
    mismatch_reason = db.Column(db.String(200))
    is_match = db.Column(db.Boolean)
    car_id = db.Column(db.Integer)

class RegisteredCarDetails(db.Model):
    __tablename__ = 'registeredcardetails'
    registration_id = db.Column(db.Integer, primary_key=True)  # Correct column name 'registration_id'
    car_model = db.Column(db.String(50))
    car_picture = db.Column(db.LargeBinary)
    color = db.Column(db.String(20))
    plate_number = db.Column(db.String(10))
    plate_origin = db.Column(db.String(20))


# Function to get the database connection
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )
    return db

# Teardown function to close the database connection
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

with app.app_context():    
    db = get_db()
    cursor = db.cursor()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        entered_username = request.form['username']
        entered_password = request.form['password']

        db = get_db()
        cursor = db.cursor()

        cursor.execute('SELECT Password FROM users WHERE Username = %s', (entered_username,))
        user = cursor.fetchone()

        if user and user[0] == entered_password:
            print("Authentication successful!")
            session['logged_in'] = True
            session['username'] = entered_username  # Set the username in the session
            return redirect(url_for('index'))
        else:
            print("Authentication failed!")

    return render_template('login.html')
    

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Protect other routes with a login check
def check_login():
    if 'logged_in' not in session:
        return redirect(url_for('login'))


#Index Page
#####################################################################################################################

@app.route('/', methods=['GET', 'POST'])
def index():
    print("Inside index function")
    if session.get('logged_in'):
        print("User logged in")
        return render_template('index.html')
    else:
        print("User not logged in, redirecting to login page")
        return redirect(url_for('login'))


#Upload file code
####################################################################################################################
@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    check_login()
    return render_template('upload_image.html')

# Color Detection
##############################################################################################################
model_color = keras.models.load_model(r'C:\Users\alaah\Desktop\Abed Thesis\My Code\Color Detection\best_model.h5')

@app.route('/predict_color', methods=['GET', 'POST'])
def predict_color():
    check_login()
    input_shape = (224, 224, 3)
    num_classes = 15

    # Load the image from the POST request using BytesIO
    file = request.files['image']
    img = load_img(BytesIO(file.read()), target_size=input_shape[:2])
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    preds = model_color.predict(x)

    # Get the predicted class with the highest confidence
    pred_class = preds.argmax(axis=-1)
    pred_prob = preds.max(axis=-1)

    class_dict = {
        'beige': 0, 'black': 1, 'blue': 2, 'brown': 3, 'gold': 4, 'green': 5, 'grey': 6,
        'orange': 7, 'pink': 8, 'purple': 9, 'red': 10, 'silver': 11, 'tan': 12, 'white': 13, 'yellow': 14
    }

    # Create a dictionary to store the confidence for each color class
    color_confidences = defaultdict(float)

    # Iterate over the predictions and store the confidence for each color
    for i, confidence in zip(pred_class, pred_prob):
        color = [k for k, v in class_dict.items() if v == i][0]
        color_confidences[color] = max(color_confidences[color], confidence)

    # Find the color with the highest confidence
    pred_color = max(color_confidences, key=color_confidences.get)
    confidence_value = round(color_confidences[pred_color], 2)

    response = {
        'color': pred_color,
        'confidence': float(confidence_value)  # Cast to float
    }
    return jsonify(response)


#playback video code
#######################################################################################################

model_plate = torch.hub.load('ultralytics/yolov5', 'custom', path=r'best.pt', force_reload=True)
model_origin = YOLO(r"C:\Users\alaah\Desktop\Abed Thesis\My Code\PlateOriginCheck\origin_best.pt")


static_folder = os.path.join(os.path.dirname(__file__), 'static')
croppedCar_images_folder = os.path.join(static_folder, 'croppedCar_images')
detectedCar_images_folder = os.path.join(static_folder, 'detectedCar_images')
croppedPlate_images_folder = os.path.join(static_folder, 'croppedPlate_images')
detectedPlate_images_folder = os.path.join(static_folder, 'detectedPlate_images')

# Load YOLOv3 model (replace this with your YOLOv3 model loading code)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("yolov3.txt", 'r') as f:
    classes_yolov3 = [line.strip() for line in f.readlines()]

# Load YOLOv5 model (replace this with your YOLOv5 model loading code)
model_yolov5 = YOLO('yolov5s.pt')

# Load YOLOv8 model
model_yolov8 = YOLO('yolov8n.pt')
names = model_yolov8.names
# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Store the track history
track_history = defaultdict(lambda: [])

def perform_ocr(image, lang='eng', psm=7, resize_factor=2.0):
    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(image, lang=lang, config=f'--psm {psm}')

    # Remove spaces and newlines
    text = re.sub(r'\s+', '', text)

    return text

def increase_brightness(image, value=30):
    """
    Increases the brightness of the input image by adding a constant value to all pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    brightened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brightened_image


def filter_dark_regions(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply Gaussian blur to reduce noise and improve thresholding
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Adaptive thresholding to handle varying lighting conditions
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert the binary image
    binary_image = cv2.bitwise_not(binary_image)

    return binary_image

color_files = {'image': None} # Initialize color_files with a default value
# Dictionary to store the sizes of previously saved car images
saved_car_sizes = {}


#convert image to Bytes to save BLOB format in the database
def image_to_bytes(image):
    _, img_bytes = cv2.imencode('.jpg', image)
    return img_bytes.tobytes()



# Create an empty list to store detected car details
detected_cars = []
ocr_text = '0'
car_image_bytes = None
plate_image_bytes = None
color_value = None
def process_frame(frame, classes, net, db):
    ocr_text = '0'
    car_image_bytes = None
    plate_image_bytes = None
    color_value = None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Assign timestamp at the beginning
    class_ids = []  # Initialize an empty list for class_ids
    confidences = []  # Initialize an empty list for confidences
    Width, Height = frame.shape[1], frame.shape[0]
    scale = 0.00392

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model_yolov8.track(frame, persist=True)

    # Check if results is None
    if results is None:
        # Handle case when no objects are detected
        print("No objects detected in the frame")
        annotated_frame = frame.copy()
    else:
        try:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu().numpy().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            annotator = Annotator(frame, line_width=2, example=names)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)

                
                # Save Cropped Car Image
                Intx = int(x - w/2)  # Calculate the starting x-coordinate of the bounding box
                Inty = int(y - h/2)  # Calculate the starting y-coordinate of the bounding box
                Intw = int(w)        # Width of the bounding box
                Inth = int(h)        # Height of the bounding box
                croppedCar_image = frame[Inty:Inty+Inth, Intx:Intx+Intw]

                # Continue with plate detection only if the car ROI is not empty
                if Intx >= 0 and Inty >= 0 and Inth >= 0 and Intw >= 0:
                    if confidences:
                        # Save the detected car image for further processing
                        car_id = track_id

                        # Insert car_id into the detectedcardetails table
                        
                        db = mysql.connector.connect(
                        host=app.config['MYSQL_HOST'],
                        user=app.config['MYSQL_USER'],
                        password=app.config['MYSQL_PASSWORD'],
                        database=app.config['MYSQL_DB']
                        )

                        cursor = db.cursor()

                        # Check if the car_id already exists in the detectedcardetails table
                        cursor.execute("SELECT COUNT(*) FROM detectedcardetails WHERE car_id = %s", (car_id,))
                        count = cursor.fetchone()[0]

                        if count == 0:
                            # Insert the row only if car_id doesn't already exist
                            cursor.execute("INSERT INTO detectedcardetails (car_id) VALUES (%s)", (car_id,))
                            print("Row with car_id =", car_id, "added to detectedcardetails table.")
                        else:
                            print("Row with car_id =", car_id, "already exists in detectedcardetails table. Skipping insertion.")

                        croppedCar_image_path = os.path.join(croppedCar_images_folder, f'cropped-Car#{car_id}.jpg')


                        # Check if the current car image size is greater than the previously saved image
                        if os.path.exists(croppedCar_image_path):
                            # Load the previously saved image
                            prev_image = cv2.imread(croppedCar_image_path)
                            prev_width = prev_image.shape[1] if prev_image is not None else 0

                            # Compare widths
                            if w > prev_width:
                                # Replace the image if the current one is wider
                                cv2.imwrite(croppedCar_image_path, croppedCar_image)
                                car_image_bytes = image_to_bytes(croppedCar_image)

                                # Update detected_car_picture for the given car_id if it exists, otherwise insert a new row
                                cursor.execute("""
                                    INSERT INTO detectedcardetails (car_id, detected_car_picture)
                                    VALUES (%s, %s)
                                    ON DUPLICATE KEY UPDATE detected_car_picture = VALUES(detected_car_picture)
                                """, (
                                    car_id,  # Car ID used for reference
                                    car_image_bytes  # BLOB data of the detected car image
                                ))
                                
                        else:
                            # If no image is saved yet, save the current one
                            cv2.imwrite(croppedCar_image_path, croppedCar_image)
                            car_image_bytes = image_to_bytes(croppedCar_image)

                            # Update detected_car_picture for the given car_id if it exists, otherwise insert a new row
                            cursor.execute("""
                                    INSERT INTO detectedcardetails (car_id, detected_car_picture)
                                    VALUES (%s, %s)
                                    ON DUPLICATE KEY UPDATE detected_car_picture = VALUES(detected_car_picture)
                                """, (
                                    car_id,  # Car ID used for reference
                                    car_image_bytes  # BLOB data of the detected car image
                                ))    
                        
                        # Load the image
                        image = cv2.imread(croppedCar_image_path)

                        # Increase brightness
                        brightened_image = increase_brightness(image)

                        # Perform color detection on the brightened image
                        color_detection_url = 'http://127.0.0.1:5000/predict_color'
                        color_files = {'image': cv2.imencode('.jpg', brightened_image)[1]}
                        color_response = requests.post(color_detection_url, files=color_files)
                        color_result = color_response.text.strip()

                        # Parse the JSON string
                        color_result = color_response.text.strip()
                        if color_result:
                            result_dict = json.loads(color_result)
                            color_value = result_dict.get("color", "Unknown")
                            confidence_value = result_dict.get("confidence", 0.0)
                        else:
                            color_value = "Unknown"
                            confidence_value = 0.0

                        print("Color Value:", color_value)
                        print("Confidence Value:", confidence_value)
                        # Update detected_color for the given car_id if it exists, otherwise insert a new row
                        cursor.execute("""
                            INSERT INTO detectedcardetails (car_id, detected_color)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE detected_color = VALUES(detected_color)
                        """, (
                            car_id,  # Car ID used for reference
                            color_value  # Color value obtained from the image processing
                        ))

                        # Commit the update
                        db.commit()

                        model_name = 'model'
                        # Update the detected car model in the detectedcardetails table
                        cursor.execute("""
                            INSERT INTO detectedcardetails (car_id, detected_car_model)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE detected_car_model = VALUES(detected_car_model)
                        """, (
                            car_id,  # Car ID used for reference
                            model_name  # Plate origin (Lebanese or Not-Lebanese)
                        ))

                        # Commit the update
                        db.commit()                
                                        

                        # Add color information to the label

                        label = f"{color_value}:{confidence_value:.2f}"

                        print("Car Bounding Box:", (Intx, Inty, Intw, Inth))
                        Textx = int(Intx)
                        Texty = int(Inty - 30)
                        cv2.putText(annotated_frame, label, (Textx, Texty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                        # Perform plate detection within the car ROI
                        if croppedCar_image is not None:
                            # Perform plate detection within the car ROI
                            image_pil = Image.fromarray(cv2.cvtColor(croppedCar_image, cv2.COLOR_BGR2RGB))
                            
                        else:
                            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            

                        plate_results = model_plate(image_pil)
                        # Process the detection results
                        processed_results = []

                        # Initialize variables for the biggest plate
                        biggest_plate_area = 0
                        biggest_plate_image = None
                        biggest_plate_box = None
                        ocr_text = None
                        filtered_plate = None

                        for detection in plate_results.xyxy[0]:
                            label = "Plate"
                            confidence = detection[-2].item()
                            # Check if the confidence is above the threshold (e.g., 50%)
                            if confidence > 0.7:
                                plate_box = detection[:4].tolist()
                                #convert to integer
                                #plate_box_int = [int(coord) for coord in plate_box]

                                #Map the coordniates
                                plate_x, plate_y, plate_w, plate_h = map(int, plate_box)

                                #real plate width
                                plate_width_for_comparision = plate_w - plate_x

                                plate_x += Intx  # Add the starting x-coordinate of the car box
                                plate_y += Inty  # Add the starting y-coordinate of the car box
                                plate_h += Inty  # Add the starting y-coordinate of the car box
                                plate_w += Intx  # Add the starting x-coordinate of the car box


                                # Crop the detected plate from the original frame
                                cropped_plate = frame[plate_y:plate_h, plate_x:plate_w]
                                

                                # Draw bounding plate_box on the original image
                                cv2.rectangle(annotated_frame, (plate_x, plate_y), (plate_w, plate_h), (255, 255, 255), 3)
                                cv2.putText(annotated_frame, f"Plate: {confidence:.2f}", (plate_x - 10, plate_y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                                biggest_plate_id = car_id
                                biggest_plate_path = os.path.join(croppedPlate_images_folder, f'Plate#{biggest_plate_id}.jpg')

                                #check if there is a previous Image
                                if os.path.exists(biggest_plate_path):
                                    # Load the previously saved image
                                    prev_plate_image = cv2.imread(biggest_plate_path)
                                    prev_plate_width = prev_plate_image.shape[1] if prev_plate_image is not None else 0
                                    prev_plate_hight = prev_plate_image.shape[0] if prev_plate_image is not None else 0
                                    prev_plate_area = prev_plate_width*prev_plate_hight

                                    # Check if current plate is bigger than the previous plate
                                    if plate_width_for_comparision > prev_plate_width:
                                        biggest_plate_image = cropped_plate
                                        biggest_plate_box = plate_box
                                        # Stretch the plate to a larger size if it's not empty
                                        stretched_plate = cv2.resize(biggest_plate_image, (1500, 800))

                                        #filter Dark Regions
                                        filtered_plate = filter_dark_regions(stretched_plate)

                                        #Replace the image if the current one is wider
                                        cv2.imwrite(biggest_plate_path, biggest_plate_image)
                                        plate_image_bytes = image_to_bytes(cropped_plate)

                                        # Update detected_plate_picture for the given car_id if it exists, otherwise insert a new row
                                        cursor.execute("""
                                            INSERT INTO detectedcardetails (car_id, detected_plate_picture)
                                            VALUES (%s, %s)
                                            ON DUPLICATE KEY UPDATE detected_plate_picture = VALUES(detected_plate_picture)
                                        """, (
                                            car_id,  # Car ID used for reference
                                            plate_image_bytes  # BLOB data of the detected plate image
                                        ))

                                        # Commit the update
                                        db.commit()
                                        

                                        # Perform OCR on the enhanced edges image
                                        
                                        ocr_text = perform_ocr(stretched_plate, lang='eng', psm=7, resize_factor=3.0)

                                        print('carID:', biggest_plate_id)
                                        print('Plate Box:', biggest_plate_box)
                                        print(plate_width_for_comparision)

                                        print('##########################################') 
                                        #perform OCR
                                        print('OCR Result:', ocr_text)   
                                        
                                        # Update detected_plate_number for the given car_id if it exists, otherwise insert a new row
                                        '''cursor.execute("""
                                            INSERT INTO detectedcardetails (car_id, detected_plate_number)
                                            VALUES (%s, %s)
                                            ON DUPLICATE KEY UPDATE detected_plate_number = VALUES(detected_plate_number)
                                        """, (
                                            car_id,  # Car ID used for reference
                                            ocr_text  # OCR result (plate number)
                                        ))
                                        # Commit the update
                                        db.commit()'''

                                        #check plate Origin
                                        print('Plate Origin Result:') 
                                        result = model_origin(croppedCar_image)
                                        cls = result[0].boxes.cls
                                        # Convert cls to a scalar value using tf.reduce_mean() or tf.reduce_sum()
                                        cls_scalar = tf.reduce_mean(cls)
                                        if cls_scalar == 1:
                                            print('Not-Lebanese Plate')
                                            origin = 'Not-Lebanese Plate'

                                        else: 
                                            print('Lebanese Plate')  
                                            origin = 'Lebanese Plate'                                      
                                        print('##########################################') 
                                        # Update detected_plate_origin for the given car_id if it exists, otherwise insert a new row
                                        cursor.execute("""
                                            INSERT INTO detectedcardetails (car_id, detected_plate_origin)
                                            VALUES (%s, %s)
                                            ON DUPLICATE KEY UPDATE detected_plate_origin = VALUES(detected_plate_origin)
                                        """, (
                                            car_id,  # Car ID used for reference
                                            origin  # Plate origin (Lebanese or Not-Lebanese)
                                        ))

                                        # Commit the update
                                        db.commit()
                                        # Call the function to compare car details
                                        print('########################Comparision will be done now')
                                        compare_car_details()
                                        print('########################it is done')

                                    else:
                                        print('WARNING: THE O LD PLATE IS BIGGER THAN THIS ONE')
                                        # Call the function to compare car details
                                        print('########################Comparision will be done now')
                                        compare_car_details()
                                        print('########################it is done')
                                else:
                                    # If no image is saved yet, save the current one
                                    print('WARNING: THIS IS THE FIRST DETECTED PLATE FOR THIS CAR')
                                    cv2.imwrite(biggest_plate_path, cropped_plate)   
                                    plate_image_bytes = image_to_bytes(cropped_plate)

                                    # Update detected_plate_picture for the given car_id if it exists, otherwise insert a new row
                                    cursor.execute("""
                                            INSERT INTO detectedcardetails (car_id, detected_plate_picture)
                                            VALUES (%s, %s)
                                            ON DUPLICATE KEY UPDATE detected_plate_picture = VALUES(detected_plate_picture)
                                        """, (
                                            car_id,  # Car ID used for reference
                                            plate_image_bytes  # BLOB data of the detected plate image
                                        ))    


                                    # Commit the update
                                    db.commit()
                                    # Call the function to compare car details
                                    print('########################Comparision will be done now')
                                    compare_car_details()
                                    print('########################it is done')

                        else:
                            print("Warning: No plate detected for the car.")

        except AttributeError:
            # Handle case when no objects are detected
            print("No objects detected in the frame")
            annotated_frame = frame.copy()
    # Return the results or use them as needed
    return annotated_frame


new_width = 480 # Adjust as needed
new_height = 860 # Adjust as needed

#Read Video frames, Skip 5 and process 1 frame
def read_frames(video_path, frame_buffer, max_frames):
    cap = cv2.VideoCapture(video_path)
    print(f"Video file {video_path} opened successfully!")

    frame_count = 0
    skip_frames = 5

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()

        if success:
            frame_count += 1

            if frame_count % skip_frames == 0:
                resized_frame = cv2.resize(frame, (new_width, new_height))
                frame_buffer.put(resized_frame)

    cap.release()


#run frame Queue Buffer, and Run Batches when needed
def gen_video(video_path, max_frames):
    frame_buffer = Queue(maxsize=40)  # Adjust buffer size as needed

    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(args.weights, args.config)

    # Function to process frames
    def process_frames(frame_buffer, result_queue):
        batch_size = 1  # or adjust as needed
        while True:
            frames = [frame_buffer.get() for _ in range(batch_size)]

            # Resize frames before processing
            resized_frames = [cv2.resize(frame, (new_width, new_height)) for frame in frames]

            # Flip frames vertically and horizontally
            flipped_frames = [cv2.flip(frame, -1) for frame in resized_frames]  # -1 for both vertical and horizontal flip

            # Process flipped frames
            processed_frames = [process_frame(frame, classes, net, db) for frame in flipped_frames]

            # Put processed frames into the result queue
            for processed_frame in processed_frames:
                result_queue.put(processed_frame)

    # Function to write frames
    def write_frames(result_queue, end_of_video):
        while True:
            processed_frame = result_queue.get()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" +
                   cv2.imencode(".jpg", processed_frame)[1].tobytes() + b"\r\n")
            if end_of_video.is_set():
                break

    result_queue = Queue(maxsize=40)  # Adjust buffer size as needed
    end_of_video = threading.Event()

    # Start reading frames in a separate thread
    read_thread = Thread(target=read_frames, args=(video_path, frame_buffer, max_frames))
    read_thread.start()

    # Start processing frames in a separate thread
    process_thread = Thread(target=process_frames, args=(frame_buffer, result_queue))
    process_thread.start()

    # Start writing frames
    frames_generator = write_frames(result_queue, end_of_video)

    return frames_generator, end_of_video

#Car Registration Page Details
##########################################################################################################

# Function to insert car details into the database
def insert_car_details(user_id, car_model, color, plate_number, plate_origin, car_picture):
    try:
        db = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )

        if db.is_connected():
            cursor = db.cursor()
            sql_query = """INSERT INTO registeredcardetails (userID, car_model, color, plate_number, plate_origin, car_picture) 
                           VALUES (%s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql_query, (user_id, car_model, color, plate_number, plate_origin, car_picture,))
            db.commit()
            print("Car details inserted successfully!")
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if db.is_connected():
            cursor.close()
            db.close()
            print("MySQL connection is closed")

# Route to handle car registration page
@app.route('/car_registration')
def car_registration():
    return render_template('car_registration.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    if 'username' in session:
        username = session['username']
        try:
            db = get_db()
            cursor = db.cursor()

            # Fetch user_id from users table where username matches the one in session
            cursor.execute('SELECT UserID FROM users WHERE username = %s', (username,))
            result = cursor.fetchone()

            if result:
                user_id = result[0]
                car_model = request.form['car_model']
                color = request.form['color']
                plate_number = request.form['plate_number']
                plate_origin = request.form['plate_origin']
                car_picture = request.files['car_picture'].read()
                
                insert_car_details(user_id, car_model, color, plate_number, plate_origin, car_picture)
                
                return "Car details submitted successfully!"
            else:
                return "User not found!"
        except Error as e:
            print("Error while connecting to MySQL", e)
        finally:
            if db.is_connected():
                cursor.close()
                db.close()
    else:
        return "User not logged in!"


#campare care details:
def compare_car_details():
    try:
        db = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )

        cursor = db.cursor()

        # Fetch rows from detectedcardetails
        cursor.execute('SELECT * FROM detectedcardetails')
        detected_rows = cursor.fetchall()

        # Iterate through detected rows
        for detected_row in detected_rows:
            # Extract relevant fields from the detected row
            detected_plate_number = detected_row[4]  # Assuming 'detected_plate_number' is the fifth column
            detected_color = detected_row[3]  # Assuming 'detected_color' is the fourth column

            # Check if plate number exists in registeredcardetails
            cursor.execute('SELECT * FROM registeredcardetails WHERE plate_number = %s', (detected_plate_number,))
            registered_row = cursor.fetchone()

            # Consume any unread results
            cursor.fetchall()

            if registered_row:
                # Plate number exists in registeredcardetails
                color_registered = registered_row[3]  # Assuming 'color' is the fourth column

                print(f"Detected Color: {detected_color}, Registered Color: {color_registered}")

                if detected_color == color_registered:
                    # Colors match, update is_match field to 'Yes' and set mismatch_reason to NULL
                    cursor.execute('UPDATE detectedcardetails SET is_match = %s, mismatch_reason = %s WHERE detected_plate_number = %s', (1, None, detected_plate_number))

                else:
                    # Colors don't match, update is_match field to 'No' and set mismatch_reason to color mismatch
                    mismatch_reason = f'color mismatch: detected {detected_color}, registered {color_registered}'
                    cursor.execute('UPDATE detectedcardetails SET is_match = %s, mismatch_reason = %s WHERE detected_plate_number = %s', (0, mismatch_reason, detected_plate_number))
                    
            else:
                # Plate number not found in registeredcardetails, update is_match field to 'No' and set mismatch_reason to unregistered plate number
                cursor.execute('UPDATE detectedcardetails SET is_match = %s, mismatch_reason = %s WHERE detected_plate_number = %s', (0, 'unregistered plate number', detected_plate_number))
                

        db.commit()
        print("Car details comparison completed!")
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL:", e)
    finally:
        # Close cursor and database connection
        if cursor:
            cursor.close()
        if db:
            db.close()
#############################################################################################################
#Violations Code 
@app.route('/violations')
def violations():
    try:
        detected_cars = DetectedCarDetails.query.all()
        registered_cars = RegisteredCarDetails.query.all()
        violations = []

        for detected_car in detected_cars:
            if not detected_car.is_match and detected_car.mismatch_reason:
                violation = {
                    'detected_car': detected_car,
                    'registered_car': next((car for car in registered_cars if car.plate_number == detected_car.detected_plate_number), None),
                    'mismatch_reason': detected_car.mismatch_reason
                }
                violations.append(violation)

        # Encode images to base64 strings
        for car in detected_cars:
            if car.detected_car_picture:
                car.detected_car_picture = base64.b64encode(car.detected_car_picture).decode('utf-8')

        for Rcar in registered_cars:
            if Rcar.car_picture:
                Rcar.car_picture = base64.b64encode(Rcar.car_picture).decode('utf-8')

        return render_template('violations.html', violations=violations, detected_cars=detected_cars, registered_cars=registered_cars)
    except Exception as e:
        print("Error:", e)
        return "An error occurred while fetching violations data"
####################################################################################################

# Route to handle registered car details page
@app.route('/registered_data')
def registered_data():
    try:
        registered_cars = RegisteredCarDetails.query.all()
        
        for Rcar in registered_cars:
            if Rcar.car_picture:
                Rcar.car_picture = base64.b64encode(Rcar.car_picture).decode('utf-8')

        return render_template('registered_data.html', registered_cars=registered_cars)
    except Exception as e:
        print("Error:", e)
        return "An error occurred while fetching data"

@app.route('/edit_car', methods=['POST'])
def edit_car():
    try:
        registration_id = request.form['registration_id']
        new_car_model = request.form['car_model']
        new_color = request.form['color']
        new_plate_number = request.form['plate_number']
        
        # Execute the SQL update query
        db = get_db()
        cursor = db.cursor()
        sql = """
        UPDATE registeredcardetails
        SET car_model = %s, color = %s, plate_number = %s
        WHERE registration_id = %s
        """
        cursor.execute(sql, (new_car_model, new_color, new_plate_number, registration_id))
        db.commit()
        cursor.close()
        
        return redirect(url_for('registered_data'))
    except Exception as e:
        print("Error:", e)
        return "An error occurred while updating car details"

@app.route('/delete_car', methods=['GET'])
def delete_car():
    try:
        registration_id = request.args.get('id')
        
        # Execute the SQL delete query
        db = get_db()
        cursor = db.cursor()
        sql = "DELETE FROM registeredcardetails WHERE registration_id = %s"
        cursor.execute(sql, (registration_id,))
        db.commit()
        cursor.close()
        
        return redirect(url_for('registered_data'))
    except Exception as e:
        print("Error:", e)
        return "An error occurred while deleting the car"

@app.route('/edit_car/<int:registration_id>', methods=['GET'])
def edit_car_form(registration_id):
    try:
        db = get_db()
        cursor = db.cursor()
        sql = "SELECT car_model, color, plate_number FROM registeredcardetails WHERE registration_id = %s"
        cursor.execute(sql, (registration_id,))
        car = cursor.fetchone()
        cursor.close()
        if car:
            return render_template('edit_car.html', car={
                'registration_id': registration_id,
                'car_model': car[0],
                'color': car[1],
                'plate_number': car[2]
            })
        else:
            return "Car not found", 404
    except Exception as e:
        print("Error:", e)
        return "An error occurred while fetching car details"
###################################################################################################

# Endpoint to process video from a file path
@app.route("/playback_video", methods=["GET", "POST"])
def video_from_file():
    check_login()
    video_path = r"C:\Users\alaah\Desktop\\Abed Thesis\My Code\Main app\static\Playback Videos\Video3.mp4"  # Replace with the actual path to your video file
    frames_generator, end_of_video = gen_video(video_path, max_frames=1000)
    return Response(frames_generator, mimetype="multipart/x-mixed-replace; boundary=frame")
########################################################################################################
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', default='yolov3.cfg', help='yolov3.cfg')
    ap.add_argument('-w', '--weights', default='yolov3.weights', help='yolov3.weights')
    ap.add_argument('-cl', '--classes', default='yolov3.txt', help='yolov3.txt')
    args = ap.parse_args()

    app.run(port=5000)