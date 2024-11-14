import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the pre-trained model
svc = pickle.load(open('trained_doctor_model.pkl', 'rb'))

# Load datasets
description_dataset = pd.read_csv("description.csv")
precaution_dataset = pd.read_csv("precautions_df.csv")
workout_dataset = pd.read_csv("workout_df.csv")
medication_dataset = pd.read_csv("medications.csv")
diet_dataset = pd.read_csv("diets.csv")

# List of symptoms and diseases
symptoms_list = {
    'itching 🤧': 0, 
    'skin_rash 🌸': 1, 
    'nodal_skin_eruptions 🌱': 2, 
    'continuous_sneezing 🤧': 3, 
    'shivering ❄️': 4, 
    'chills 🥶': 5, 
    'joint_pain 🤕': 6, 
    'stomach_pain 🤢': 7, 
    'acidity 🔥': 8, 
    'ulcers_on_tongue 🤒': 9, 
    'muscle_wasting 💪': 10, 
    'vomiting 🤮': 11, 
    'burning_micturition 🔥💧': 12, 
    'spotting_urination 💧': 13, 
    'fatigue 😴': 14, 
    'weight_gain ⚖️': 15, 
    'anxiety 😟': 16, 
    'cold_hands_and_feets 👐❄️': 17, 
    'mood_swings 😤😨': 18, 
    'weight_loss ⚖️⤵️': 19, 
    'restlessness 🏃‍♂️': 20, 
    'lethargy 😓': 21, 
    'patches_in_throat 🦷': 22, 
    'irregular_sugar_level 🍬': 23, 
    'cough 🤧': 24, 
    'high_fever 🌡️': 25, 
    'sunken_eyes 👀': 26, 
    'breathlessness 😤': 27, 
    'sweating 💦': 28, 
    'dehydration 💧❌': 29, 
    'indigestion 🍔💨': 30, 
    'headache 🤕': 31, 
    'yellowish_skin 🟡': 32, 
    'dark_urine 🟤': 33, 
    'nausea 🤢': 34, 
    'loss_of_appetite ❌🍴': 35, 
    'pain_behind_the_eyes 👀🤕': 36, 
    'back_pain 🦴🤕': 37, 
    'constipation 🚫💩': 38, 
    'abdominal_pain 🤰🤕': 39, 
    'diarrhoea 💩💦': 40, 
    'mild_fever 🌡️': 41, 
    'yellow_urine 🟡💧': 42, 
    'yellowing_of_eyes 🟡👀': 43, 
    'acute_liver_failure 🍷💔': 44, 
    'fluid_overload 💧➡️': 45, 
    'swelling_of_stomach 🤰': 46, 
    'swelled_lymph_nodes 🦋': 47, 
    'malaise 😓': 48, 
    'blurred_and_distorted_vision 👓': 49, 
    'phlegm 💨': 50, 
    'throat_irritation 🦷': 51, 
    'redness_of_eyes 👀🔴': 52, 
    'sinus_pressure 🌬️': 53, 
    'runny_nose 🌬️💧': 54, 
    'congestion 🏠': 55, 
    'chest_pain 💔': 56, 
    'weakness_in_limbs 💪😔': 57, 
    'fast_heart_rate ❤️‍🔥': 58, 
    'pain_during_bowel_movements 💩🤕': 59, 
    'pain_in_anal_region 💩❌': 60, 
    'bloody_stool 💩🩸': 61, 
    'irritation_in_anus 💩😤': 62, 
    'neck_pain 🦴🤕': 63, 
    'dizziness 🌀': 64, 
    'cramps 🤕': 65, 
    'bruising 🩸': 66, 
    'obesity ⚖️🥴': 67, 
    'swollen_legs 🦵💦': 68, 
    'swollen_blood_vessels 💉🔴': 69, 
    'puffy_face_and_eyes 😴💧': 70, 
    'enlarged_thyroid 🦋': 71, 
    'brittle_nails 💅': 72, 
    'swollen_extremeties 🦵🔴': 73, 
    'excessive_hunger 🍔🍟': 74, 
    'extra_marital_contacts 💑': 75, 
    'drying_and_tingling_lips 💋': 76, 
    'slurred_speech 🗣️': 77, 
    'knee_pain 🦵🤕': 78, 
    'hip_joint_pain 🦵🤕': 79, 
    'muscle_weakness 💪😔': 80, 
    'stiff_neck 🦴🤕': 81, 
    'swelling_joints 🦴💥': 82, 
    'movement_stiffness 🦵🚶‍♂️': 83, 
    'spinning_movements 🌀': 84, 
    'loss_of_balance ⚖️❌': 85, 
    'unsteadiness 🤸‍♂️': 86, 
    'weakness_of_one_body_side 🦵': 87, 
    'loss_of_smell 👃❌': 88, 
    'bladder_discomfort 💧❌': 89, 
    'foul_smell_of_urine 💧💩': 90, 
    'continuous_feel_of_urine 💧🕰️': 91, 
    'passage_of_gases 💨': 92, 
    'internal_itching 💧🤧': 93, 
    'toxic_look_(typhos) 🍷💔': 94, 
    'depression 😞': 95, 
    'irritability 😡': 96, 
    'muscle_pain 💪🤕': 97, 
    'altered_sensorium 🧠❌': 98, 
    'red_spots_over_body 🔴': 99, 
    'belly_pain 🤰🤕': 100, 
    'abnormal_menstruation 💧⚠️': 101, 
    'dischromic_patches 🌈': 102, 
    'watering_from_eyes 👁️💧': 103, 
    'increased_appetite 🍔🍕': 104, 
    'polyuria 💧💧': 105, 
    'family_history 👨‍👩‍👧‍👦': 106, 
    'mucoid_sputum 💨': 107, 
    'rusty_sputum 💨🟤': 108, 
    'lack_of_concentration 🧠❌': 109, 
    'visual_disturbances 👀❌': 110, 
    'receiving_blood_transfusion 🩸': 111, 
    'receiving_unsterile_injections 💉': 112, 
    'coma 🛌❌': 113, 
    'stomach_bleeding 💔💧': 114, 
    'distention_of_abdomen 🤰': 115, 
    'history_of_alcohol_consumption 🍷': 116, 
    'fluid_overload.1 💧➡️': 117, 
    'blood_in_sputum 🩸💨': 118, 
    'prominent_veins_on_calf 💉': 119, 
    'palpitations ❤️‍🔥': 120, 
    'painful_walking 🦵🚶‍♂️': 121, 
    'pus_filled_pimples 💥': 122, 
    'blackheads 🖤': 123, 
    'scurring 💥': 124, 
    'skin_peeling 🌿': 125, 
    'silver_like_dusting 🌨️': 126, 
    'small_dents_in_nails 💅': 127, 
    'inflammatory_nails 💅🔥': 128, 
    'blister 💥': 129, 
    'red_sore_around_nose 👃🔴': 130, 
    'yellow_crust_ooze 🟡💧': 131
}



disease_list = {15: "Fungal infection",4: "Allergy",16: "GERD",9: "Chronic cholestasis",14: "Drug Reaction",33: "Peptic ulcer diseae",1: "AIDS",12: "Diabetes", 17: "Gastroenteritis",6: "Bronchial Asthma",23: "Hypertension",30: "Migraine",7: "Cervical spondylosis",32: "Paralysis (brain hemorrhage)",28: "Jaundice",29: "Malaria",8: "Chicken pox",11: "Dengue",37: "Typhoid",40: "hepatitis A",19: "Hepatitis B",20: "Hepatitis C",21: "Hepatitis D",22: "Hepatitis E",3: "Alcoholic hepatitis",36: "Tuberculosis",10: "Common Cold",34: "Pneumonia",13: "Dimorphic hemmorhoids(piles)",18: "Heart attack",39: "Varicose veins",26: "Hypothyroidism",24: "Hyperthyroidism",25: "Hypoglycemia",31: "Osteoarthristis",5: "Arthritis",0: "(vertigo) Paroymsal  Positional Vertigo",2: "Acne",38: "Urinary tract infection",35: "Psoriasis",27: "Impetigo",}

# Helper function to get the disease details
def helper(dis):
    desc = description_dataset[description_dataset['Disease'] == dis]["Description"]
    desc = " ".join([w for w in desc])

    prec = precaution_dataset[precaution_dataset["Disease"] == dis][['Precaution_1','Precaution_2','Precaution_3', 'Precaution_4']]
    prec = [p for p in prec.values]

    work = workout_dataset[workout_dataset['disease'] == dis]['workout']
    work = [w for w in work.values]

    medi = medication_dataset[medication_dataset['Disease'] == dis]['Medication']
    medi = [m for m in medi.values]

    diet = diet_dataset[diet_dataset['Disease'] == dis]['Diet']
    diet = [w for w in diet.values]

    return desc, prec, work, medi, diet

# Streamlit user interface
def main():

    st.set_page_config(page_title="AI Doctor Web", page_icon="🏥")
    st.title('Disease Prediction App')

    # Select symptoms
    st.subheader("Select Symptoms (you can select multiple symptoms):")
    selected_symptoms = st.multiselect("Choose symptoms", symptoms_list.keys())

    if selected_symptoms:  # Proceed if symptoms are selected
        # Map the selected symptoms to 132 features
        input_features = np.zeros(len(symptoms_list))
        for symptom in selected_symptoms:
            input_features[symptoms_list[symptom]] = 1

        # Predict the disease
        disease = svc.predict([input_features])[0]
        # st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>Predicted Disease: {disease_list[disease]}</h2>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background-color: rgba(204, 0, 0, 6); border-radius: 5px; padding: 10px; text-align: center;'>
            <h2 style='color: rgba(255, 99, 71, 1); font-size: 19px;'>Predicted Disease: {disease_list[disease]}</h2>
        </div>
    """, unsafe_allow_html=True)

        # Get disease details
        desc, prec, work, medi, diet = helper(disease_list[disease])

        # Display details
        # Display the disease information with a more elegant layout and emojis
        st.markdown(f"<div style='background-color: rgba(23,45,67,1);margin-top:30px; border-radius: 10px; padding: 5px; text-align: center;'>"
        f"<h4 style='color: #4682B4;'>Description</h4><p style='font-size: 16px; color: rgba(200, 230, 255, 1);'>{desc}</p></div>", unsafe_allow_html=True)

# Display Precautions with a check for data availability
        if prec and len(prec[0]) > 0:
            st.markdown(f"<div style='background-color: rgba(227, 191, 45, 4); margin-top:30px;border-radius: 5px; padding: 5px; text-align: center;'><h4 style='color: #4B4B00;font-weight:700;'>⚠️ Precautions ⚠️</h4></div>",unsafe_allow_html=True)
            for p in prec[0]:
                st.markdown(f"<p style='font-size: 16px;'>{'⚠️ ' + str(p)}</p>", unsafe_allow_html=True)
        else:
            st.write("No specific precautions available for this disease.")

# Display Recommended Workouts with a check
        if work:
            st.markdown(f"<div style='background-color: rgba(45, 167, 45, 1);margin-top:30px; border-radius: 5px; padding: 5px; text-align: center;'><h3 style='color: rgba(0, 104, 0, 1)'>💪 Recommended Workouts 🏋️‍♂️</h3></div>", unsafe_allow_html=True)
            for w in work:
                st.markdown(f"<p style='font-size: 16px;'>{'💪 '+ str(w)}</p>", unsafe_allow_html=True)
        else:
            st.write("No specific workouts recommended for this disease.")

# Display Medications with a check

        if medi:
            st.markdown(f"<div style='background-color: rgba(120, 45, 167, 1);margin-top:30px; border-radius: 5px; padding: 5px; text-align: center;'><h3 style='color: rgba(75, 0, 130, 1)'>🏥 Medication 🏥</h3></div>", unsafe_allow_html=True)
            for m in medi:
                st.markdown(f"<p style='font-size: 16px;'>{'💊 '+ str(m)}</p>", unsafe_allow_html=True)
        else:
            st.write("No specific workouts recommended for this disease.")
        




# Display Dietary Recommendations with a check
        if diet:
            st.markdown(f"<div style='background-color:rgba(255, 165, 0, 1);margin-top:30px; border-radius: 5px; padding: 5px; text-align: center;'><h3 style='color:  rgba(204, 85, 0, 1)'>🥣 Diets 🥣</h3></div>", unsafe_allow_html=True)
            for d in diet:
                st.markdown(f"<p style='font-size: 16px;'>{'🥗 '+ str(d)}</p>", unsafe_allow_html=True)
        else:
            st.write("No specific workouts recommended for this disease.")



    else:
        st.info("Please select symptoms to get a prediction.")

if __name__ == '__main__':
    main()
