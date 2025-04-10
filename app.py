import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import cv2
import statistics
import sqlite3
import base64
from fpdf import FPDF
from datetime import datetime
import os

# Function to connect to the SQLite database
def connect_db():
    try:
        connection = sqlite3.connect('users.db')
        return connection
    except Exception as e:
        return None

# Function to create a new user account
def create_account(name, email, number, age, gender, password):
    try:
        connection = connect_db()
        cursor = connection.cursor()

        query = "INSERT INTO users (name, email, number, age, gender, password) VALUES (?, ?, ?, ?, ?, ?)"
        cursor.execute(query, (name, email, number, age, gender, password))
        connection.commit()

        st.success("Account created successfully!")
        
        # Redirect to login page after successful sign-up
        st.session_state['menu'] = 'Home'
        
    except Exception as e:
        st.error(f"Error creating account: {e}")
    finally:
        cursor.close()
        connection.close()

# Function to check user login credentials
def check_login(email, password):
    connection = connect_db()
    if connection is not None:
        try:
            cursor = connection.cursor()
            query = "SELECT * FROM users WHERE email=? AND password=?"
            cursor.execute(query, (email, password))
            user = cursor.fetchone()
            if user:
                st.session_state['logged_in'] = True
                st.session_state['user_name'] = user[1]  # Assuming 'name' is the second field
                st.session_state['user_email'] = user[2]  # Assuming 'email' is the third field
                
                st.success("Login successful!")
                
                # Redirect to the predict page after successful login
                st.session_state['menu'] = 'Predict'
                
                return user
            else:
                st.error("Invalid credentials.")
        except Exception as e:
            st.error(f"Error logging in: {e}")
        finally:
            connection.close()

# Function to create an appointment
def create_appointment(name, date, time, email):
    try:
        connection = connect_db()
        cursor = connection.cursor()

        query = "INSERT INTO appointments (name, date, time, email) VALUES (?, ?, ?, ?)"
        cursor.execute(query, (name, date, time, email))
        connection.commit()

        st.success("Appointment created successfully!")
    except Exception as e:
        st.error(f"Error creating appointment: {e}")
    finally:
        cursor.close()
        connection.close()

# Function to check if an image is an ECG image with valid ECG waves
def is_ecg_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            return False  # If the image couldn't be loaded
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check the aspect ratio of the image (ECG reports are usually landscape)
        height, width = gray_image.shape
        if width < height:
            return False
        
        # Check for typical ECG grid patterns (Hough Line Transform)
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None or len(lines) < 20:
            return False  # Not enough lines detected for a typical ECG grid pattern
        
        # Check for edges (ECG reports should have many vertical and horizontal edges)
        edge_density = np.sum(edges) / (height * width)
        if edge_density < 0.05:  # Adjust the threshold based on testing
            return False
        
        # Additional checks for ECG-like characteristics (e.g., line continuity)
        # Further analysis of wave-like patterns or specific ECG features can be added here
        
        return True  # If all checks pass, it's likely an ECG image

    except Exception as e:
        st.error(f"Error in ECG image validation: {e}")
        return False

# Function to extract metadata from an ECG image
def extract_ecg_metadata(image_path):
    try:
        ecg_image = cv2.imread(image_path)
        resized_image = cv2.resize(ecg_image, (2213, 1572))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(binary_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        st_elevation_values = []
        pathological_q_waves_values = []
        t_wave_inversions_values = []
        abnormal_qrs_complexes_values = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 20:
                st_elevation_values.append(h)
            if w > 10 and h < 5:
                pathological_q_waves_values.append(h)
            if h < 10 and w > 15:
                abnormal_qrs_complexes_values.append(h)
            if h < 10 and w < 10:
                t_wave_inversions_values.append(h)

        def calculate_stats(values):
            return {
                'max': max(values) if values else 0,
                'mean': statistics.mean(values) if values else 0,
                'median': statistics.median(values) if values else 0
            }

        metadata = {
            'Max ST Elevation (Height)': calculate_stats(st_elevation_values)['max'],
            'Mean ST Elevation (Height)': calculate_stats(st_elevation_values)['mean'],
            'Median ST Elevation (Height)': calculate_stats(st_elevation_values)['median'],
            'Max Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['max'],
            'Mean Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['mean'],
            'Median Pathological Q Wave (Height)': calculate_stats(pathological_q_waves_values)['median'],
            'Max T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['max'],
            'Mean T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['mean'],
            'Median T Wave Inversion (Height)': calculate_stats(t_wave_inversions_values)['median'],
            'Max Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['max'],
            'Mean Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['mean'],
            'Median Abnormal QRS Complex (Height)': calculate_stats(abnormal_qrs_complexes_values)['median']
        }

        return metadata
    except Exception as e:
        st.error(f"Error extracting ECG metadata: {e}")
        return {}

# Function to predict disease based on ECG metadata and store in the report column of the user
def predict_disease(image_path, model, scaler, label_encoder, user_email):
    if not is_ecg_image(image_path):
        return "Invalid image. Please upload a valid ECG image."
    
    metadata = extract_ecg_metadata(image_path)
    if not metadata:
        return "Error: Could not extract metadata from the image."

    metadata_df = pd.DataFrame([metadata])

    if scaler:
        try:
            metadata_scaled = scaler.transform(metadata_df)
            prediction_index = model.predict(metadata_scaled)[0]
            predicted_class = label_encoder.inverse_transform([prediction_index])[0]

            # Store the predicted disease in the user's report column
            try:
                connection = connect_db()
                cursor = connection.cursor()
                update_query = "UPDATE users SET report = ? WHERE email = ?"
                cursor.execute(update_query, (predicted_class, user_email))
                connection.commit()
                st.success(f"The predicted disease has been saved to your account: {predicted_class}")
            except Exception as e:
                st.error(f"Error updating report in database: {e}")
            finally:
                cursor.close()
                connection.close()

            return predicted_class
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return "Prediction failed."
    else:
        return "Error: Scaler not available."

# Function to get precautions based on predicted disease type
def get_precautions(disease_type):
    precautions = {
        "myocardial": [
            "1. Take prescribed medications as directed.",
            "2. Avoid heavy physical exertion.",
            "3. Monitor heart rate and report any irregularities.",
            "4. Follow up with a cardiologist regularly."
        ],
        "historyofmi": [
            "1. Maintain a healthy lifestyle with a balanced diet.",
            "2. Exercise regularly but avoid strenuous activities.",
            "3. Regular check-ups with a healthcare provider.",
            "4. Keep track of any new symptoms."
        ],
        "abnormal": [
            "1. Follow up with a healthcare provider for further evaluation.",
            "2. Monitor for any changes in symptoms.",
            "3. Maintain a healthy lifestyle."
        ],
        "normal": [
            "1. Maintain a healthy diet rich in fruits, vegetables, and whole grains.",
            "2. Engage in regular physical activity, such as walking or cycling.",
            "3. Avoid smoking and limit alcohol intake.",
            "4. Get regular health check-ups and monitor blood pressure regularly."
        ]
    }
    return precautions.get(disease_type.lower(), ["No specific precautions available."])

# Function to generate a PDF report
def generate_pdf_report(user_name, predicted_class, metadata, precautions):
    try:
        pdf = FPDF()
        pdf.add_page()

        # Add a border around the entire page (A4 sheet)
        pdf.set_fill_color(255, 255, 255)  # White background
        pdf.set_draw_color(0, 0, 0)  # Black border color
        pdf.rect(5.0, 5.0, 200.0, 287.0)  # Rectangular border (A4 sheet size minus margins)

        # Set Times New Roman font for title
        pdf.set_font("Times", size=16, style='B')
        pdf.cell(200, 10, txt="Predictza Report", ln=True, align='C')

        # Add user details
        pdf.set_font("Times", size=13, style='B')
        pdf.cell(200, 10, txt=f"Name: {user_name}", ln=True)

        # Highlight the predicted class with a bold font and colored background
        pdf.set_font("Times", size=14, style='B')
        pdf.set_fill_color(173, 216, 230)  # Light blue background color
        pdf.set_text_color(0, 0, 0)  # Black text color
        pdf.cell(190, 10, txt=f"PREDICTED DISEASE TYPE: {predicted_class.upper()}", ln=True, border=1, align='C', fill=True)
        pdf.ln(10)

        # Highlight and add border for Metadata section
        pdf.set_font("Times", size=12, style='B')
        pdf.set_fill_color(255, 228, 225)  # Light pink background color
        pdf.set_text_color(0, 0, 0)  # Black text color
        pdf.cell(190, 10, txt="Metadata:", ln=True, border=1, fill=True)
        pdf.ln(5)

        # Print metadata with borders around each line
        pdf.set_font("Times", size=12)
        for key, value in metadata.items():
            pdf.cell(190, 10, txt=f"{key}: {value}", ln=True, border=1)
        pdf.ln(10)

        # Highlight and add border for Precautions section
        pdf.set_font("Times", size=12, style='B')
        pdf.set_fill_color(255, 228, 225)  # Light pink background color
        pdf.cell(190, 10, txt="Precautions:", ln=True, border=1, fill=True)
        pdf.ln(5)

        # Print precautions with borders around each line
        pdf.set_font("Times", size=12)
        for precaution in precautions:
            pdf.cell(190, 10, txt=precaution, ln=True, border=1)

        # Save the PDF file
        pdf_output = f"{user_name}_prediction_report.pdf"
        pdf.output(pdf_output)

        return pdf_output
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

# Load pre-trained models and other assets
try:
    rf_model = joblib.load('rf_model.joblib')
    xgb_model = joblib.load('xgb_model.joblib')
    cat_model = joblib.load('cat_model.joblib')
    voting_clf = joblib.load('voting_clf.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Streamlit app setup
def main():
    # Initialize session state if not already done
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'menu' not in st.session_state:
        st.session_state['menu'] = 'Home'  # Default page
    
    menu = ["Home", "Sign Up", "Predict", "Appointments", "About"]
    
    # Control page flow based on session state
    if st.session_state['logged_in']:
        st.session_state['menu'] = 'Predict'
    
    choice = st.sidebar.selectbox("Menu", menu, index=menu.index(st.session_state['menu']))
    st.markdown(
    """
    <h1 style='text-align: center; font-family: "Times New Roman", Times, serif; font-size: 50px;'>
        PREDICTZA
    </h1>
    """,
    unsafe_allow_html=True)

    if choice == "Home":
        st.session_state['menu'] = 'Home'
        st.markdown(
            """
            <h2 style='text-align: center; font-family: "Times New Roman", Times, serif; font-size: 30px;'>
                Know Your Cardiac Health
            </h2>
            """,
            unsafe_allow_html=True)
        st.subheader("Login to Your Account")
        
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user = check_login(email, password)
            if user:
                st.write(f"Welcome to Predictza App")
    
    elif choice == "Sign Up":
        st.session_state['menu'] = 'Sign Up'
        st.subheader("Create a New Account")
        
        name = st.text_input("Name", placeholder="Enter your name")
        email = st.text_input("Email", placeholder="Enter your email")
        number = st.text_input("Phone Number", placeholder="Enter your phone number")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Create Account"):
            create_account(name, email, number, age, gender, password)
    
    elif choice == "Predict":
        st.session_state['menu'] = 'Predict'
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            st.subheader("Upload ECG Image for Prediction")
        
            user_name = st.text_input("Enter your name", value=st.session_state.get('user_name', ''))
            image_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
            
            if st.button("Predict"):
                if image_file is not None:
                    # Save the uploaded image file
                    image_path = f"./temp_{datetime.now().timestamp()}.png"
                    with open(image_path, "wb") as f:
                        f.write(image_file.getbuffer())
                    
                    # Make a prediction and store it in the user's report column
                    predicted_class = predict_disease(image_path, voting_clf, scaler, label_encoder, st.session_state.get('user_email', ''))
                    st.write(f"Predicted Disease: {predicted_class}")
                    
                    # Get metadata and precautions
                    metadata = extract_ecg_metadata(image_path)
                    precautions = get_precautions(predicted_class)
                    
                    st.write("Precautions:")
                    for precaution in precautions:
                        st.write(precaution)
                    
                    # Generate and provide a PDF report
                    pdf_report = generate_pdf_report(user_name, predicted_class, metadata, precautions)
                    with open(pdf_report, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        b64 = base64.b64encode(pdf_bytes).decode()
                        download_link = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_report}">Download Report</a>'
                        st.markdown(download_link, unsafe_allow_html=True)

    elif choice == "Appointments":
        st.session_state['menu'] = 'Appointments'
        st.subheader("Book an Appointment")

        name = st.text_input("Name")
        email = st.text_input("Email")
        date = st.date_input("Appointment Date")
        time = st.time_input("Appointment Time")

        if st.button("Book Appointment"):
            create_appointment(name, str(date), str(time), email)
    
    elif choice == "About":
        st.session_state['menu'] = 'About'
        st.markdown(
            """
            <h2 style='font-family: "Times New Roman", Times, serif;'>
                About Our App
            </h2>
            <p style='font-family: "Times New Roman", Times, serif; font-size: 20px;'>
                This app aims to create a machine learning model using ensemble learning techniques to predict heart disease types from ECG reports. The solution involves preprocessing data, training models, combining them into a voting classifier, and providing diagnosis results.
            </p>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()