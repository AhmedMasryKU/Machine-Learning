from numpy import genfromtxt
import csv
import numpy as np
import math

def read_csv(filename):
    with open(filename, newline='') as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

def gaussian(x,mean, std):
    var = float(std)**2
    return math.exp(-(float(x)-float(mean))**2/(2*var)) / (2*math.pi*var)**.5

def safe_log(num):
    return math.log(num + 1e-12)


# Predict function.
def predict(images, male_prob, female_prob, male_means, female_means, male_std, female_std):
    predictions = []
    for im in images:
        male_p = safe_log(male_prob)
        female_p = safe_log(female_prob)
        for i in range(len(male_means)):
            male_p += safe_log(gaussian(im[i], male_means[i], male_std[i]))
            female_p += safe_log(gaussian(im[i], female_means[i], female_std[i]))

        predictions.append(1 if male_p > female_p else 2)
    return predictions

def confusion_matrix(predictions, correct_result):
    conf = np.zeros((2,2))
    for i in range(len(correct_result)):
        if correct_result[i] == 1:
            if predictions[i] == 1:
                conf[0][1] += 1
            else:
                conf[0][0] += 1
        else:
            if predictions[i] == 2:
                conf[1][0] += 1
            else:
                conf[1][1] += 1
    return conf

def main():
    #load data
    input_data = np.array(read_csv('hw01_images.csv'))
    output_data = np.array(read_csv('hw01_labels.csv'))

    # Split data into training and testing
    training_input_data = input_data[0:200]
    training_output_data = output_data[0:200]
    testing_input_data = input_data[200:400]
    testing_output_data = output_data[200:400]

    # Calculating prior distribution probabilities.
    male_prob = (training_output_data[training_output_data==2]).shape[0] / len(training_output_data)
    female_prob = (training_output_data[training_output_data==1]).shape[0] / len(training_input_data)

    # Calculating means of features.
    male_means = np.mean(training_input_data[np.where(training_output_data == 2)[0]], axis=0)
    female_means = np.mean(training_input_data[np.where(training_output_data == 1)[0]], axis=0)
    print(female_means)
    # Calculating standard deviations.
    male_std = np.std(training_input_data[np.where(training_output_data == 2)[0]], axis=0)
    female_std = np.std(training_input_data[np.where(training_output_data == 1)[0]], axis=0)

    # Calculate confusion matrices.
    training_set_conf_mat = confusion_matrix(predict(training_input_data, male_prob, female_prob, male_means, female_means, male_std, female_std),  training_output_data)
    testing_set_conf_mat = confusion_matrix(predict(testing_input_data, male_prob, female_prob, male_means, female_means, male_std, female_std),  testing_output_data)

    print("Training set confusion matrix.")
    print(training_set_conf_mat)
    print("Testing set confusion matrix.")
    print(testing_set_conf_mat)

if __name__=='__main__':
    main()