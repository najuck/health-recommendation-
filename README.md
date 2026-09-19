# AI Health & Medicine Recommendation System

An end-to-end Machine Learning web application that predicts possible health conditions based on reported symptoms and delivers comprehensive clinical recommendations including **disease descriptions, precautions, medications, diets, and active recovery workouts**.

---

## 📌 Project Overview & Highlights

- **Machine Learning Core**: Multi-class **Support Vector Classifier (SVC)** trained on **132 clinical symptom features** across **41 disease classes**.
- **Fault-Tolerant Input Parsing**: Normalizes user input by handling casing variations, spaces, underscores, and colloquial aliases (e.g. *coughing* &rarr; *cough*, *fever* &rarr; *high_fever*).
- **Comprehensive 360° Recommendations**:
  - 🩺 **Diagnosis & Medical Description**
  - 🛡️ **Preventative Measures & Precautions**
  - 💊 **Common Over-the-Counter & Prescription Medications**
  - 🥗 **Dietary Guidelines & Nutrition Advice**
  - 🏃 **Workouts & Physical Activity Recommendations**
- **Modern Responsive Web Interface**: Built with **Flask** and **Bootstrap 5.3**, featuring interactive symptom search, quick-select chips, direct report cards, and modal dialogs.
- **Production-Ready & Robust**: Zero-crash architecture with full protection against `KeyError` and `IndexError` on edge cases.

---

## 🛠️ Tech Stack & Architecture

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn (Support Vector Machines / SVC), NumPy, Pandas
- **Web Framework**: Flask, Jinja2
- **Frontend**: Bootstrap 5.3, Bootstrap Icons, Modern Responsive CSS
- **Model Serialization**: Pickle

---

## 📂 Project Structure

```
health-recommendation-/
├── models/
│   └── svc.pkl                       # Serialized trained SVC model
├── templates/
│   ├── index.html                    # Main diagnosis & recommendations view
│   ├── about.html                    # Project background & methodology
│   ├── blog.html                     # Technical architecture write-up
│   ├── developer.html                # Developer profile & ML stack
│   └── contact.html                  # Feedback & inquiries
├── static/
│   └── healthcare.webp               # App branding & logo assets
├── Training.csv.csv                  # Model training dataset
├── symtoms_df.csv                    # Symptom mappings
├── description.csv                   # Disease medical descriptions
├── precautions_df.csv                # 4-step precautions per condition
├── medications.csv                   # Recommended medications
├── diets.csv                         # Dietary guidelines
├── workout_df.csv                    # Physical recovery & lifestyle habits
├── main.py                           # Flask application backend
├── requirements.txt                  # Pinned dependencies
├── run.bat                           # One-click Windows runner
└── README.md                         # Project documentation
```

---

## 🚀 Quick Start & Installation

### Option 1: One-Click Launch (Windows)
Double-click `run.bat`. It will automatically activate the local environment and start the web server on `http://127.0.0.1:5000/`.

### Option 2: Manual Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/najuck/health-recommendation-.git
   cd health-recommendation-
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

5. **Open in your browser**:
   Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧪 Testing & Verification

Run the test suite to verify route stability and model predictions:

```bash
python -c "from main import app; c = app.test_client(); print('Home status:', c.get('/').status_code); print('Predict status:', c.post('/predict', data={'symptoms': 'itching, skin_rash'}).status_code)"
```

---

## ⚠️ Medical Disclaimer

*This application is developed strictly for educational and research demonstration purposes. It does not replace professional medical diagnosis, advice, or treatment. Users should always consult qualified healthcare providers regarding medical conditions.*
