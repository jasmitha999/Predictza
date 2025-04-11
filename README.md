# Predictza
Predictza is a Streamlit-based web application that predicts heart disease types using ECG images. The app leverages ensemble learning models to analyze ECG reports and provide diagnostic predictions along with preventive recommendations.

🩺 Features
User authentication (Sign up / Log in)

ECG image upload and validation

Extraction of key ECG metadata

Disease prediction using ensemble models (Random Forest, XGBoost, CatBoost)

Personalized report generation in PDF format

Appointment booking system

Downloadable prediction report with metadata and precautions

🧠 Technologies Used
Frontend/UI: Streamlit

Backend: Python

ML Models: scikit-learn, XGBoost, CatBoost

Image Processing: OpenCV

Database: SQLite

PDF Generation: FPDF

📦 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/predictzaaiml.git
cd predictzaaiml
Create a virtual environment and activate it:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
📁 Project Structure
bash
Copy
Edit
predictzaaiml-main/
│
├── app.py                  # Main application logic
├── rf_model.joblib         # Random Forest model
├── xgb_model.joblib        # XGBoost model
├── cat_model.joblib        # CatBoost model
├── voting_clf.joblib       # Ensemble Voting Classifier
├── scaler.joblib           # Scaler used for preprocessing
├── label_encoder.joblib    # Encodes class labels
├── users.db                # SQLite database for user and appointments
└── requirements.txt        # Project dependencies
📝 Usage
Sign up with your details or log in if you already have an account.

Upload an ECG image for prediction.

View disease type, metadata analysis, and suggested precautions.

Download a comprehensive report in PDF format.

Book appointments for further medical consultation.

🔐 Security
User data is stored locally in an SQLite database. Always ensure secure hosting practices if deploying this app online.

📄 License
This project is licensed under the MIT License.

