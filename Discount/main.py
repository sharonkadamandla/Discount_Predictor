
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def predictDisease(symptoms):

    # Reading the train.csv by removing the 
    # last column since it's an empty column
    DATA_PATH = r"C:\Users\ajayk\OneDrive\Documents\GitHub\Disease_Predictor\Discount\disease\Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis=1)

    # Encoding the target value into numerical
    # value using LabelEncoder
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])

    X = data.iloc[:, :-1]


    final_rf_model = pickle.load(open(r"C:\Users\ajayk\OneDrive\Documents\GitHub\Disease_Predictor\Discount\models\final_rf_model.pkl", "rb"))
    final_nb_model = pickle.load(open(r"C:\Users\ajayk\OneDrive\Documents\GitHub\Disease_Predictor\Discount\models\final_nb_model.pkl", "rb"))
    final_svm_model = pickle.load(open(r"C:\Users\ajayk\OneDrive\Documents\GitHub\Disease_Predictor\Discount\models\final_svm_model.pkl", "rb"))

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptoms = symptoms.split(",")
    symptom_index = {symptom.strip(): i for i, symptom in enumerate(X.columns)}

    data_dict = {
        "symptom_index": symptom_index,
        "predictions_classes": encoder.classes_
    }

    # Creating input data for the model
    input_data = np.zeros(len(data_dict["symptom_index"]))
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.strip())
        if index is not None:
            input_data[index] = 1

    # Reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)

    # Generating individual outputs
    rf_prediction = encoder.inverse_transform([final_rf_model.predict(input_data)])[0]
    nb_prediction = encoder.inverse_transform([final_nb_model.predict(input_data)])[0]
    svm_prediction = encoder.inverse_transform([final_svm_model.predict(input_data)])[0]

    # Making final prediction by taking mode of all predictions
    all_predictions = [rf_prediction, nb_prediction, svm_prediction]
    final_prediction = np.unique(all_predictions)[0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    formatted_predictions = "\n".join([f"{key}: {value}" for key, value in predictions.items()])

    return formatted_predictions







