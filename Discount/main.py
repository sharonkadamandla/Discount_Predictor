import numpy as np
import pickle
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


def predictDisease(input_data):
    
    final_rf_model = pickle.load(open("final_rf_model.pkl", "rb"))
    final_nb_model = pickle.load(open("final_nb_model.pkl", "rb"))
    final_svm_model = pickle.load(open("final_svm_model.pkl", "rb"))
    symptom=input_data
    encoder = LabelEncoder()

    #change X to the input that you are getting
    #symptoms = X.columns.values
 
    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}

    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split(",")])
        symptom_index[symptom] = index
 
    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }
    

    symptoms = symptoms.split(",")

    # creating input data for the model
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
    formatted_predictions = "\n".join([f"{key}: {value}" for key, value in predictions.items()])

    return formatted_predictions



