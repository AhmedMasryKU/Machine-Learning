from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

# Hyper parameters. 
eta = 0.0005
epsilon = 1e-3
max_iteration = 500
H = 20

def read_csv(filename):
    with open(filename, newline='') as f_input:
        data = []
        i=0
        for row in csv.reader(f_input):
            if i == 0:
                i = 1
                continue
            data.append(list(map(float, row)))
        return data

# Get Borders
def get_borders(bin_width, min_val, max_val, border_type = "left"):
    borders = np.arange(min_val, max_val, bin_width)
    if border_type == "right":
        borders += bin_width
    return borders

# Regrossomer.
def Regresogram(left_borders, right_borders, data_train, data_train_eruptions,  data_train_waiting):
    out = []
    for i in range(len(left_borders)):
        s = 0
        num = 0
        for j in range(len(data_train)):
            if data_train_eruptions[j] >= left_borders[i] and data_train_eruptions[j] < right_borders[i]:
                s += data_train_waiting[j]
                num += 1
        out.append(s/num)
    return out

def RMSE_Regressomer(x, correct_y, estimated_y, left_borders, right_borders):
    error = 0
    for i in range(len(correct_y)):
        estimated = 0.0
        for j in range(len(left_borders)):
            if x[i] >= left_borders[j] and x[i] < right_borders[j]:
                estimated = estimated_y[j]
        diff = (correct_y[i] - estimated)**2
        error += diff
    error /= len(correct_y)
    return math.sqrt(error)

def mean_smoother(data_intervals, data_train, bin_width, data_train_eruptions, data_train_waiting):
    out = []
    for i in range(len(data_intervals)):
        num = 0.0
        den = 0.0
        for j in range(len(data_train)):
            if data_train_eruptions[j] > data_intervals[i] - 0.5 * bin_width and data_train_eruptions[j] <= data_intervals[i] + 0.5 * bin_width:
                num += data_train_waiting[j]
                den += 1
        out.append(num/den)
    return out

def RMSE_smooth(x, correct_y,  estimated_y, data_intervals, bin_width, data_train, data_train_waiting, data_train_eruptions):
    error = 0.0
    for i in range(len(correct_y)):
        estimated = 0.0
        num = 0.0
        den = 0.0
        for j in range(len(data_train)):
            if x[i] > data_train_eruptions[j] - 0.5*bin_width and x[i] <= data_train_eruptions[j] + 0.5*bin_width:
                num += data_train_waiting[j]
                den += 1
        estimated = num / den
        diff = (correct_y[i] - estimated)**2
        error += diff
    error /= len(correct_y)
    return math.sqrt(error)

# Kernel
def kernel(data_intervals, data_train, data_train_eruptions, data_train_waiting, bin_width):
    out = []
    for i in range(len(data_intervals)):
        num = 0.0
        den = 0.0
        for j in range(len(data_train)):
            num += (1 / math.sqrt(2*math.pi)) * math.exp(-0.5 * (data_intervals[i] - data_train_eruptions[j])**2 / bin_width**2) * data_train_waiting[j]
            den += (1 / math.sqrt(2*math.pi)) * math.exp(-0.5 * (data_intervals[i] - data_train_eruptions[j])**2 / bin_width**2)
        out.append(num/den)
    return out

# Kernel RMSE.
def RMSE_kernel(x, correct_y,  estimated_y, data_intervals, data_train_eruptions, data_train_waiting, bin_width):
    error = 0.0
    for i in range(len(correct_y)):
        estimated = 0.0

        num = 0.0
        den = 0.0
        for j in range(len(data_train_eruptions)):
            num += (1 / math.sqrt(2*math.pi)) * math.exp(-0.5 * (x[i] - data_train_eruptions[j])**2 / bin_width**2) * data_train_waiting[j]
            den += (1 / math.sqrt(2*math.pi)) * math.exp(-0.5 * (x[i] - data_train_eruptions[j])**2 / bin_width**2)
        
        estimated = num/den
        diff = (correct_y[i] - estimated)**2
        error += diff
    
    error /= len(correct_y)
    return math.sqrt(error)


def main():
    # Load data. 
    data = np.array(read_csv("hw04_data_set.csv"))

    # Split the data. 
    data_train = data[0:150]
    data_test = data[150:272]

    data_train_eruptions = [i[0] for i in data_train]
    data_train_waiting = [i[1] for i in data_train]

    data_test_eruptions = [i[0] for i in data_test]
    data_test_waiting = [i[1] for i in data_test]

    # Regresogram paraneters. 
    bin_width = 0.37
    min_val = 1.5
    max_val = 5.2

    # Borders. 
    left_borders = get_borders(bin_width, min_val, max_val, border_type = "left")
    right_borders = get_borders(bin_width, min_val, max_val, border_type = "right")

    # Regrosssogram. output. 
    hist_out = Regresogram(left_borders, right_borders, data_train, data_train_eruptions,  data_train_waiting)

    # Plot Regrossogram. 
    plt.scatter(data_train_eruptions, data_train_waiting, label= "Training")
    plt.scatter(data_test_eruptions, data_test_waiting, label ="Test")

    borders= np.append(left_borders, right_borders[len(right_borders)-1])
    hist = np.append(hist_out, hist_out[len(hist_out)-1])
    plt.step(borders, hist, where = 'post', color = 'black')

    plt.ylabel('Waiting time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title(label = ("h = " + str(bin_width)))
    plt.legend()
    plt.show()

    # RMSE of Ressogram. 
    r_rmse = RMSE_Regressomer(data_test_eruptions, data_test_waiting, hist_out, left_borders, right_borders)
    print("Regressogram => RMSE is ", r_rmse, "when h is ", bin_width)

    # Mean Smoother. 
    data_intervals = np.arange(min_val, max_val, 0.01)
    smooth_out = mean_smoother(data_intervals, data_train, bin_width, data_train_eruptions, data_train_waiting)

    # Plot Meas smoother. 
    plt.scatter(data_train_eruptions, data_train_waiting, label= "Training")
    plt.scatter(data_test_eruptions, data_test_waiting, label ="Test")
    plt.step(data_intervals, smooth_out, where = 'post', color = 'black')
    plt.ylabel('Waiting time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title(label = ("h = " + str(bin_width)))
    plt.legend()
    plt.show()

    # Smooth RMSE.
    smooth_rmse = RMSE_smooth(data_test_eruptions, data_test_waiting, smooth_out, data_intervals, bin_width, data_train, data_train_waiting, data_train_eruptions)
    print("Running Mean Smoother => RMSE is",  smooth_rmse, "when h is",  bin_width)
    
    # Kernel. 
    kernel_out = kernel(data_intervals, data_train, data_train_eruptions, data_train_waiting, bin_width)

    # Kernel plot. 
    plt.scatter(data_train_eruptions, data_train_waiting, label= "Training")
    plt.scatter(data_test_eruptions, data_test_waiting, label ="Test")
    plt.step(data_intervals, kernel_out, where = 'post', color = 'black')
    plt.ylabel('Waiting time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title(label = ("h = " + str(bin_width)))
    plt.legend()
    plt.show()

    # Kernel rmse. 
    kernel_rmse = RMSE_kernel(data_test_eruptions, data_test_waiting, kernel_out, data_intervals, data_train_eruptions, data_train_waiting, bin_width)
    print("Kernel Smoother => RMSE is ", kernel_rmse , "when h is ", bin_width)

if __name__=='__main__':
    main()