import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
train_data = pd.read_csv("Titanic_train.csv")
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
import joblib
import re 

model = joblib.load("Logistic_model.pkl")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import re

# Load data and model
train_data = pd.read_csv("Titanic_train.csv")
model = joblib.load("Logistic_model.pkl")
st.title("Titanic Logistic Regression Model")


def extract_title(name: str) -> str:
    """Extract a title (Mr, Mrs, Miss, etc.) from a full name string.
    Falls back to 'Mr' when no title can be found.
    """
    if not isinstance(name, str) or name.strip() == "":
        return "Mr"
    match = re.search(r',\s*([^.]*)\.', name)
    if match:
        return match.group(1).strip()
    match2 = re.search(r'\b(Mr|Mrs|Miss|Master|Dr|Rev|Col|Major|Mlle|Mme|Ms|Sir|Lady|Capt|Don|Jonkheer)\b', name)
    if match2:
        return match2.group(1)
    return "Mr"


# Build a title encoder from training data (the model was trained with an encoded
# 'Name' column containing titles encoded as integers)
train_titles = train_data['Name'].apply(extract_title).astype(str)
title_le = LabelEncoder()
title_le.fit(train_titles)


with st.form("titanic_form"):
    passengerid = st.number_input("Passenger ID", 1, 10000, 1)

    pclass_options = {
        "First Class": 1,
        "Second Class": 2,
        "Third Class": 3,
    }
    pclass_name = st.selectbox("Passenger Class", list(pclass_options.keys()))

    sex = st.radio("Sex", ["Male", "Female"])
    age = st.slider("Age", 0, 100, 25)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
    parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
    Name = st.text_input("Name (Full name as in training data)")
    fare = st.number_input('Fare', 0.0, 500.0, 50.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    submit = st.form_submit_button("Predict")


if submit:
    pclass_num = pclass_options[pclass_name]
    sex_num = 0 if sex == 'Male' else 1
    # encode embarked using the training data categories
    embarked_le = LabelEncoder()
    embarked_le.fit(train_data['Embarked'].astype(str))
    embarked_num = int(embarked_le.transform([embarked])[0])

    # Encode the Name/title the model expects (it was trained with a numeric 'Name' feature)
    title = extract_title(Name) if Name else 'Mr'
    if title not in title_le.classes_:
        fallback = pd.Series(train_titles).mode()[0] if len(train_titles) > 0 else 'Mr'
        title = fallback
    name_encoding = int(title_le.transform([title])[0])

    input_data = pd.DataFrame([{
        'PassengerId': passengerid,
        'Pclass': pclass_num,
        'Name': name_encoding,
        'Sex': sex_num,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked_num,
    }], dtype='object')

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    st.subheader("Prediction Result")
    st.success(f"Prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}")
    st.info(f"Survival Probability: {probability[0]:.2f}")