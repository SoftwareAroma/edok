import pickle
import warnings
import pandas as pd
import numpy as np
import gradio as gr
import streamlit as st
from sklearn.model_selection import train_test_split
from keras.src.saving import load_model

warnings.filterwarnings("ignore")

# Load the saved models
ann_model = load_model('models/ann_model.keras')

# Load the label encoder
with open('models/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the dataset and preprocess it
df = pd.read_csv("data/dataset.csv")
for col in df.columns:
    df[col] = df[col].str.replace('_', ' ')

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
df = df.fillna(0)

df1 = pd.read_csv('data/Symptom-severity.csv')
df1['Symptom'] = df1['Symptom'].str.replace('_', ' ')

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

d = pd.DataFrame(vals, columns=cols)
d = d.replace('dischromic patches', 0).infer_objects(copy=False)
d = d.replace('spotting urination', 0).infer_objects(copy=False)
df = d.replace('foul smell of urine', 0).infer_objects(copy=False)

data = df.iloc[:, 1:].values  # Exclude the 'Disease' column
labels = df['Disease'].values

# Ensure consistent lengths
assert len(data) == len(labels), "Data and labels must have the same number of samples"

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=42)


# Function to predict possible diseases based on symptoms
def predict_possible_diseases(symptoms_list, top_n=5) -> list[tuple]:
    # Load symptom severity data
    symptom_severity_df = pd.read_csv('data/Symptom-severity.csv')
    symptom_severity_df['Symptom'] = symptom_severity_df['Symptom'].str.replace('_', ' ')

    # Create a zeroed input array
    input_data = np.zeros((1, x_train.shape[1]))

    # Encode the symptoms
    for symptom in symptoms_list:
        severity = symptom_severity_df[symptom_severity_df['Symptom'] == symptom]['weight'].values[0]
        col_index = df.columns.get_loc('Symptom_' + str(symptoms_list.index(symptom) + 1))
        input_data[0, col_index - 1] = severity  # -1 to shift since df.columns[0] is 'Disease'

    # Predict the disease probabilities
    pred_prob = ann_model.predict(input_data)

    # Get top N diseases with the highest probabilities
    top_indices = np.argsort(pred_prob[0])[::-1][:top_n]
    top_diseases = label_encoder.inverse_transform(top_indices)
    top_probabilities = pred_prob[0][top_indices]

    # Return the top N diseases and their probabilities
    return list(zip(top_diseases, top_probabilities))


# Define Gradio interface
def gradio_interface(symptoms: str) -> object:
    symptoms_list = [s.strip() for s in symptoms.split(',')]
    return predict_possible_diseases(symptoms_list, top_n=5)


# Create the Gradio interface
examples: list[list[str]] = [
    ["itching, skin rash, nodal skin eruptions"],
    ["chest pain, phlegm, runny nose, high fever, throat irritation, congestion, redness of eyes"],
]

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter symptoms separated by commas..."),
    outputs="json",
    title="Disease Prediction from Symptoms",
    description="Enter symptoms separated by commas to predict possible diseases along with their probabilities.",
    examples=examples
)

# Launch the Gradio interface
iface.launch(share=True)
