import os
import numpy as np
from scipy.fft import fft
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/aryanmehrotra/Desktop/College/Masters/Fall_2023/EECE5699/Labs/Lab 4/lab4-a-mehrotra/handout/')
from extractKeyStroke import extractKeyStroke
from sklearn.neural_network import MLPClassifier

classifier_inputs = []

# Generate threshold values for training data and transforms samples to frequency domain
def get_KeyPress_TrainingData(training_files, path):
    f = open("/Users/aryanmehrotra/Desktop/College/Masters/Fall_2023/EECE5699/Labs/Lab 4/lab4-a-mehrotra/submission/thresh_holds.txt", "w")

    for file in training_files:
        file_name = path + file
        print(file_name)
        threshold_val = 17
        while True:
            peaks, numClicks, keys = extractKeyStroke(file_name, 100, threshold_val)
            # Training data has 100 key presses
            if not (len(peaks) == 100):
                threshold_val = threshold_val - 0.5

            else:
                f.write(str(threshold_val) + "\n")
                # print(threshold_val)
                f.flush()
                break
            
        pushPeak_fft = abs(fft(peaks))

        for i in range(len(pushPeak_fft)):
            classifier_inputs.append(pushPeak_fft[i])
    
    f.close()
    return classifier_inputs

# Generate threshold values for test data and transforms samples to frequency domain
def get_KeyPress_TestData(test_file):
    threshold_val = 17
    while True:
        peaks, numClicks, keys = extractKeyStroke(test_file, 8, threshold_val)
        # Test data has 8 key presses
        if not (len(peaks) == 8):
            threshold_val = threshold_val - 0.5

        else:
            break
            
    pushPeak_fft = abs(fft(peaks))

    password = []
    for j in range(len(pushPeak_fft)):
        password.append(pushPeak_fft[j])
    return password

# Checks the accuracy of the neural net by running test data through it, prints table with results
def accuracy_Check(mlp_model, file_path, test_files):
    tempNum = 0
    for file in test_files:
        file_name = file_path + file
        #print(file_name)
        test_data = get_KeyPress_TestData(file_name)
        prediction = mlp_model.predict_proba(test_data)

        first = 0
        second = 0
        third = 0

        axis = np.array(np.argmax(prediction, axis=1))

        for j in range(8):
            axis_unsorted = list(prediction[j])
            axis_sorted = np.array(prediction[j])
            axis_sorted = np.sort(axis_sorted)[::-1]

            if axis[j] == tempNum:
                first += 1
            elif axis_unsorted.index(axis_sorted[1]) == tempNum:
                second += 1
            elif axis_unsorted.index(axis_sorted[2]) == tempNum:
                third += 1

        tempNum += 1
        print("Test " + file + ": First:  " + str(first) + " Second:  " + str(second) + " Third:  " + str(third))

# Accuracy check for the secret in the secret audio files
def get_Top3_Accuracy(secret: list, mlp: MLPClassifier):
    passwords = mlp.predict_proba(secret)

    first = ""
    second = ""
    third = ""

    for j in range(8):
        axis_unsorted = list(passwords[j])
        axis_sorted = np.array(passwords[j])
        axis_sorted = np.sort(axis_sorted)[::-1]

        for i in range(26):
            if axis_unsorted.index(axis_sorted[0]) == i:
                first = first + chr(97 + i)
            elif axis_unsorted.index(axis_sorted[1]) == i:
                second = second + chr(97 + i)
            elif axis_unsorted.index(axis_sorted[2]) == i:
                third = third + chr(97 + i)

    print("first letter choice:   " + first)
    print("second letter choice:  " + second)
    print("third letter choice:   " + third + "\n")

# Helper function called to check passwords
def password_Check(mlp_model, file_path, secret_files):
    for file in secret_files:
        file_name = file_path + file
        secret = get_KeyPress_TestData(file_name)
        print(file_name + '\n')
        password = mlp_model.predict(secret)
        get_Top3_Accuracy(secret, mlp_model)

# Saves the model
def save_Model(model):
    filename = 'keyboard.sav'
    pickle.dump(model, open(filename, 'wb'))

# Loads in the model
def load_Model():
    filename = 'keyboard.sav'
    loadedModel = pickle.load(open(filename, 'rb'))
    return loadedModel

# Checks the secret and test files
def test_Attack_Password():
    mlp_model = load_Model()
    # Data File Store
    audio_data = '/Users/aryanmehrotra/Desktop/College/Masters/Fall_2023/EECE5699/Labs/Lab 4/lab4-a-mehrotra/handout/data/'
    # Create File Lists
    all_files = os.listdir(audio_data)
    test_files = [file for file in all_files if 'test' in file.lower()]
    secret_files = [file for file in all_files if 'secret' in file.lower()]
    # Sort the files in alphabetical order
    test_files.sort()
    secret_files.sort()

    password_Check(mlp_model, audio_data, secret_files)
    accuracy_Check(mlp_model, audio_data, test_files)

# Generates threshold data and trains the neural net
def generate_MLP_Model():
    # Data File Store
    audio_data = '/Users/aryanmehrotra/Desktop/College/Masters/Fall_2023/EECE5699/Labs/Lab 4/lab4-a-mehrotra/handout/data/'
    # Create File Lists
    all_files = os.listdir(audio_data)
    training_files = [file for file in all_files if 'secret' not in file.lower() and 'test' not in file.lower()]
    # Sort the files in alphabetical order
    training_files.sort()

    inputs = get_KeyPress_TrainingData(training_files, audio_data)
    labels = []
    for i in range(26):
        for j in range(100):
            labels.append(chr(97 + i))
    
    mlp_model = MLPClassifier(hidden_layer_sizes=150, max_iter=3000, verbose=True)
    mlp_model.fit(inputs, labels)

    save_Model(mlp_model)

# Function calls
generate_MLP_Model()
test_Attack_Password()
    