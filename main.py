import os
import ast
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, redirect, url_for, jsonify

# Setup base directory for robust path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets safely
precautions = pd.read_csv(os.path.join(BASE_DIR, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(BASE_DIR, "workout_df.csv"))
description = pd.read_csv(os.path.join(BASE_DIR, "description.csv"))
medications = pd.read_csv(os.path.join(BASE_DIR, "medications.csv"))
diets = pd.read_csv(os.path.join(BASE_DIR, "diets.csv"))

# Strip whitespace from disease column in all datasets for 100% reliable matching
precautions['Disease'] = precautions['Disease'].astype(str).str.strip()
workout['disease'] = workout['disease'].astype(str).str.strip()
description['Disease'] = description['Disease'].astype(str).str.strip()
medications['Disease'] = medications['Disease'].astype(str).str.strip()
diets['Disease'] = diets['Disease'].astype(str).str.strip()

# Load trained model
model_path = os.path.join(BASE_DIR, "models", "svc.pkl")
svc = pickle.load(open(model_path, 'rb'))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Model feature names (132 exact features trained in SVC)
feature_names = list(svc.feature_names_in_)

# Mapping of diseases (all leading/trailing whitespaces removed)
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    0: '(vertigo) Paroymsal Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis',
    27: 'Impetigo', 5: 'Arthritis', 31: 'Osteoarthristis', 25: 'Hypoglycemia'
}

def normalize_symptom_string(s):
    """Normalize input string to lowercase, remove punctuation, replace spaces/dashes with underscores."""
    return s.strip().lower().replace('-', '_').replace(' ', '_')

# Build comprehensive lookup dictionary for symptom matching
symptom_canonical_map = {}
for feat in feature_names:
    norm = normalize_symptom_string(feat)
    symptom_canonical_map[norm] = feat
    symptom_canonical_map[feat.lower().strip()] = feat
    symptom_canonical_map[feat.replace('_', ' ').lower().strip()] = feat

# Common colloquial synonyms and variations
symptom_aliases = {
    'coughing': 'cough',
    'fever': 'high_fever',
    'high fever': 'high_fever',
    'mild fever': 'mild_fever',
    'sneezing': 'continuous_sneezing',
    'vomit': 'vomiting',
    'itchy': 'itching',
    'itch': 'itching',
    'rash': 'skin_rash',
    'rashes': 'skin_rash',
    'diarrhea': 'diarrhoea',
    'loose_motions': 'diarrhoea',
    'loose motions': 'diarrhoea',
    'running_nose': 'runny_nose',
    'running nose': 'runny_nose',
    'tiredness': 'fatigue',
    'tired': 'fatigue',
    'exhaustion': 'fatigue',
    'dizzy': 'dizziness',
    'nauseous': 'nausea',
    'nauseated': 'nausea',
    'cramping': 'cramps',
    'gas': 'passage_of_gases',
    'gases': 'passage_of_gases',
    'belly_ache': 'belly_pain',
    'belly ache': 'belly_pain',
    'head_ache': 'headache',
    'head ache': 'headache',
    'body_ache': 'muscle_pain',
    'body ache': 'muscle_pain',
    'body_pain': 'muscle_pain',
    'body pain': 'muscle_pain',
    'breath_problem': 'breathlessness',
    'shortness_of_breath': 'breathlessness',
    'shortness of breath': 'breathlessness',
    'difficulty_breathing': 'breathlessness',
    'stomach_ache': 'stomach_pain',
    'stomach ache': 'stomach_pain',
    'foul_smell_ofurine': 'foul_smell_of urine',
    'spotting_urination': 'spotting_ urination',
    'dischromic_patches': 'dischromic _patches'
}

for alias, target in symptom_aliases.items():
    norm_alias = normalize_symptom_string(alias)
    if target in symptom_canonical_map:
        canonical = symptom_canonical_map[target]
        symptom_canonical_map[norm_alias] = canonical
        symptom_canonical_map[alias.lower().strip()] = canonical

# List of clean human-readable symptom names for datalist/autocomplete
readable_symptoms = sorted(list(set(
    f.replace('.', '').replace('_', ' ').strip().title() for f in feature_names
)))

####################### Helper Functions #######################

def helper(dis):
    """Retrieve descriptions, precautions, medications, diet, and workouts safely."""
    dis_clean = dis.strip().lower()

    # Description
    desc_rows = description[description['Disease'].str.lower() == dis_clean]['Description']
    desc = " ".join([str(w) for w in desc_rows.values]) if len(desc_rows) > 0 else "Detailed description currently being updated."

    # Precautions
    prec_rows = precautions[precautions['Disease'].str.lower() == dis_clean][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = []
    if len(prec_rows) > 0:
        for val in prec_rows.values[0]:
            if pd.notna(val) and str(val).strip():
                pre.append(str(val).strip().capitalize())

    # Medications (parse stringified list safely)
    med_rows = medications[medications['Disease'].str.lower() == dis_clean]['Medication']
    med = []
    if len(med_rows) > 0:
        for item in med_rows.values:
            try:
                parsed = ast.literal_eval(str(item))
                if isinstance(parsed, list):
                    med.extend([str(x).strip() for x in parsed])
                else:
                    med.append(str(item).strip())
            except Exception:
                med.append(str(item).strip())

    # Diets (parse stringified list safely)
    diet_rows = diets[diets['Disease'].str.lower() == dis_clean]['Diet']
    die = []
    if len(diet_rows) > 0:
        for item in diet_rows.values:
            try:
                parsed = ast.literal_eval(str(item))
                if isinstance(parsed, list):
                    die.extend([str(x).strip() for x in parsed])
                else:
                    die.append(str(item).strip())
            except Exception:
                die.append(str(item).strip())

    # Workouts
    workout_rows = workout[workout['disease'].str.lower() == dis_clean]['workout']
    wrkout = [str(w).strip() for w in workout_rows.values if pd.notna(w)]

    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    """Predict disease using a DataFrame with exact feature names."""
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    
    for symptom in patient_symptoms:
        if symptom in input_df.columns:
            input_df.loc[0, symptom] = 1
        # Also ensure duplicate column fluid_overload.1 is populated if fluid_overload is present
        if symptom == 'fluid_overload' and 'fluid_overload.1' in input_df.columns:
            input_df.loc[0, 'fluid_overload.1'] = 1

    prediction_id = svc.predict(input_df)[0]
    return diseases_list.get(prediction_id, "Unknown Condition")

####################### Web Routes #######################

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', readable_symptoms=readable_symptoms)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))

    symptoms_raw = request.form.get('symptoms', '').strip()
    if not symptoms_raw:
        return render_template('index.html', 
                               readable_symptoms=readable_symptoms,
                               error="Please enter at least one symptom.")

    # Split by comma or newline
    raw_tokens = [s.strip() for s in symptoms_raw.replace('\n', ',').split(',') if s.strip()]
    
    matched_features = []
    unmatched_tokens = []

    for token in raw_tokens:
        clean_token = token.strip("[]'\" ")
        norm_token = normalize_symptom_string(clean_token)
        
        if norm_token in symptom_canonical_map:
            matched_features.append(symptom_canonical_map[norm_token])
        elif clean_token.lower() in symptom_canonical_map:
            matched_features.append(symptom_canonical_map[clean_token.lower()])
        else:
            unmatched_tokens.append(clean_token)

    if not matched_features:
        return render_template('index.html',
                               readable_symptoms=readable_symptoms,
                               entered_symptoms=symptoms_raw,
                               error=f"Could not match any symptoms from: '{', '.join(unmatched_tokens)}'. Please select from the suggested symptoms.")

    # Predict condition
    predicted_disease = get_predicted_value(matched_features)
    desc, pre, med, die, wrkout = helper(predicted_disease)

    # Human-readable recognized symptoms
    display_matched = [f.replace('.', '').replace('_', ' ').strip().title() for f in matched_features]

    return render_template('index.html',
                           readable_symptoms=readable_symptoms,
                           entered_symptoms=symptoms_raw,
                           recognized_symptoms=display_matched,
                           unrecognized_symptoms=unmatched_tokens,
                           predicted_disease=predicted_disease,
                           dis_des=desc,
                           dis_pre=pre,
                           dis_med=med,
                           dis_wrkout=wrkout,
                           dis_diet=die)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
@app.route('/Contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
@app.route('/Blog')
def blog():
    return render_template('blog.html')

@app.route('/developer')
@app.route('/Developer')
def developer():
    return render_template('developer.html')

@app.route('/api/symptoms')
def api_symptoms():
    """API endpoint returning available symptoms for autocomplete."""
    return jsonify(readable_symptoms)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
