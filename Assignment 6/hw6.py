from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import random
import torch
from torch import nn
import datetime


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
    
# Define Model.
class MLP(torch.nn.Module):
    def __init__(self, IDs_len, REGs_len):
            super(MLP, self).__init__()

            self.I_branch = nn.Sequential(
                nn.Linear(IDs_len, 64),
                nn.ReLU()
            )
            self.R_branch = nn.Sequential(
                nn.Linear(REGs_len, 48),
                nn.ReLU()
            )
            self.D_branch = nn.Sequential(
                nn.Linear(31, 48),
                nn.ReLU()
            )
            self.M_branch = nn.Sequential(
                nn.Linear(12, 32),
                nn.ReLU()
            )
            self.Y_branch = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU()
            )
            self.T_branch = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU()
            )
            ######
            self.time_branch = nn.Sequential(
                nn.Linear(96, 128),
                nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU()                
            )
            self.DW_branch = nn.Sequential(
                nn.Linear(7, 32),
                nn.ReLU()
                #nn.Dropout(0.5),
                #nn.Linear(64, 32),
                #nn.ReLU()  
            )
            #####

            self.main_branch = nn.Sequential(
                nn.Linear(224, 256),
                nn.ReLU(),
                #nn.Dropout(0.2),
                #nn.Linear(512, 128),
                #nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(), 
                nn.Linear(128, 32),
                nn.ReLU(), 
                nn.Linear(32, 1)
            )

    def forward(self, I, R, D, M, Y, T, DW):

            I_out = self.I_branch(I)
            R_out = self.R_branch(R)

            D_out = self.D_branch(D)
            M_out = self.M_branch(M)
            Y_out = self.Y_branch(Y)
            DW_out = self.DW_branch(DW)

            time_cat = torch.cat((D_out, M_out, Y_out), dim=1)
            time_out = self.time_branch(time_cat)

            T_out = self.T_branch(T)

            out_cat = torch.cat((I_out, R_out, T_out, time_out, DW_out), dim=1)
            output = self.main_branch(out_cat)
            return output
    
def main():
    # Loading the training and testing data.
    training_data = read_csv("training_data.csv")
    testing_data = read_csv("test_data.csv")

    # Extracting the number of Unique IDs and Regs 
    # Test the IDs. 
    IDs = []
    REGs = []
    for instance in training_data:
        if not instance[0] in IDs:
            IDs.append(instance[0])
        if not instance[1] in REGs:
            REGs.append(instance[1])
    for instance in testing_data:
        if not instance[0] in IDs:
            IDs.append(instance[0])
        if not instance[1] in REGs:
            REGs.append(instance[1])
            
    # Getting the Mins and Maxs for each feature. 
    Mins = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
    Maxs = [0, 0, 0, 0, 0, 0, 0]
    for instance in training_data:
        for k in range(len(instance)):
            if Mins[k] > instance[k]:
                Mins[k] = instance[k]
            if Maxs[k] < instance[k]:
                Maxs[k] = instance[k]
    # Normalize the output feature only. 
    training_data_normalized = []
    for instance in training_data:
        normalized_instance = []
        for k in range(len(instance)):
            if k==6:
                normalized_instance.append((instance[k]-Mins[k])/(Maxs[k]- Mins[k]))
            else:
                normalized_instance.append(instance[k])
        training_data_normalized.append(normalized_instance)

    # Split the data into training and validation (In my code, test_data refers to validation data.)
    random.shuffle(training_data_normalized)
    train_data = training_data_normalized[:40000]
    test_data = training_data_normalized[40000:]
    #train_data = []
    #test_data = []
    #random.shuffle(training_data_normalized)
    #for ins in training_data_normalized:
    #    if ins[4] == 2019 and ins[3] > 4 and ins[2] > 20:
    #        test_data.append(ins)
    #    else:
    #        train_data.append(ins)

    batch_size = 64
    
    # Put training data into Batches. 
    train_data_input_batches_I_vec = []
    train_data_input_batches_R_vec = []
    train_data_input_batches_D_vec = []
    train_data_input_batches_M_vec = []
    train_data_input_batches_Y_vec = []
    train_data_input_batches_T_vec = []
    train_data_input_batches_DW_vec = []

    train_data_output_batches = []

    input_batch_I_vec = []
    input_batch_R_vec = []
    input_batch_D_vec = []
    input_batch_M_vec = []
    input_batch_Y_vec = []
    input_batch_T_vec = []
    input_batch_DW_vec = []
    output_batch = []

    for i in range(1, len(train_data)+1):
        I, R, D, M, Y, TRX_T, TRX_C = train_data[i-1]

        ## To be deleted.
        #if TRX_T == 2:
        #    continue
        ##
        # Making one hot vectors. 
        I_vec = [0 for j in range(len(IDs))]
        I_vec[IDs.index(I)] = 1
        R_vec = [0 for j in range(len(REGs))]
        R_vec[REGs.index(R)] = 1
        D_vec = [0 for j in range(31)]
        D_vec[int(D)-1] = 1
        M_vec = [0 for j in range(12)]
        M_vec[int(M)-1] = 1
        Y_vec = [0, 0]
        if Y == 2019:
            Y_vec[1] = 1
        else:
            Y_vec[0] = 1
        T_vec = [0,0]
        T_vec[int(TRX_T)-1] = 1

        DW_vec = [0 for j in range(7)]
        d = datetime.datetime(int(Y), int(M), int(D))
        DW_vec[d.weekday()] = 1

        input_batch_I_vec.append(I_vec)
        input_batch_R_vec.append(R_vec)   
        input_batch_D_vec.append(D_vec)
        input_batch_M_vec.append(M_vec)
        input_batch_Y_vec.append(Y_vec)
        input_batch_T_vec.append(T_vec)
        input_batch_DW_vec.append(DW_vec)

        output_batch.append(TRX_C)

        if i % batch_size == 0 or i==len(train_data):
            train_data_input_batches_I_vec.append(input_batch_I_vec)
            train_data_input_batches_R_vec.append(input_batch_R_vec)
            train_data_input_batches_D_vec.append(input_batch_D_vec)
            train_data_input_batches_M_vec.append(input_batch_M_vec)
            train_data_input_batches_Y_vec.append(input_batch_Y_vec)
            train_data_input_batches_T_vec.append(input_batch_T_vec)
            train_data_input_batches_DW_vec.append(input_batch_DW_vec)


            train_data_output_batches.append(output_batch)

            input_batch_I_vec = []
            input_batch_R_vec = []
            input_batch_D_vec = []
            input_batch_M_vec = []
            input_batch_Y_vec = []
            input_batch_T_vec = []
            input_batch_DW_vec = []
            output_batch = []
    
    # Put test data into Batches. 
    test_data_input_batches_I_vec = []
    test_data_input_batches_R_vec = []
    test_data_input_batches_D_vec = []
    test_data_input_batches_M_vec = []
    test_data_input_batches_Y_vec = []
    test_data_input_batches_T_vec = []
    test_data_input_batches_DW_vec = []
    test_data_output_batches = []

    input_batch_I_vec = []
    input_batch_R_vec = []
    input_batch_D_vec = []
    input_batch_M_vec = []
    input_batch_Y_vec = []
    input_batch_T_vec = []
    input_batch_DW_vec = []
    output_batch = []

    for i in range(1, len(test_data)+1):
        I, R, D, M, Y, TRX_T, TRX_C = test_data[i-1]

        ## To be deleted.
        #if TRX_T == 2:
        #    continue
        ##

        # Making one hot vectors. 
        I_vec = [0 for j in range(len(IDs))]
        I_vec[IDs.index(I)] = 1
        R_vec = [0 for j in range(len(REGs))]
        R_vec[REGs.index(R)] = 1
        D_vec = [0 for j in range(31)]
        D_vec[int(D)-1] = 1
        M_vec = [0 for j in range(12)]
        M_vec[int(M)-1] = 1
        Y_vec = [0, 0]
        if Y == 2019:
            Y_vec[1] = 1
        else:
            Y_vec[0] = 1
        T_vec = [0,0]
        T_vec[int(TRX_T)-1] = 1

        DW_vec = [0 for j in range(7)]
        d = datetime.datetime(int(Y), int(M), int(D))
        DW_vec[d.weekday()] = 1

        input_batch_I_vec.append(I_vec)
        input_batch_R_vec.append(R_vec)   
        input_batch_D_vec.append(D_vec)
        input_batch_M_vec.append(M_vec)
        input_batch_Y_vec.append(Y_vec)
        input_batch_T_vec.append(T_vec)
        input_batch_DW_vec.append(DW_vec)

        output_batch.append(TRX_C)

        if i % batch_size == 0 or i==len(test_data):
            test_data_input_batches_I_vec.append(input_batch_I_vec)
            test_data_input_batches_R_vec.append(input_batch_R_vec)
            test_data_input_batches_D_vec.append(input_batch_D_vec)
            test_data_input_batches_M_vec.append(input_batch_M_vec)
            test_data_input_batches_Y_vec.append(input_batch_Y_vec)
            test_data_input_batches_T_vec.append(input_batch_T_vec)
            test_data_input_batches_DW_vec.append(input_batch_DW_vec)

            test_data_output_batches.append(output_batch)

            input_batch_I_vec = []
            input_batch_R_vec = []
            input_batch_D_vec = []
            input_batch_M_vec = []
            input_batch_Y_vec = []
            input_batch_T_vec = []
            input_batch_DW_vec = []

            output_batch = []
    # Make model instance.
    model = MLP(len(IDs), len(REGs))
    # Move to CUDA for faster training. 
    if torch.cuda.is_available():
        model = model.cuda()
    # Loss function and optimizer.
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00035)
   
    # Training part.
    model.train()
    epochs = 15
    training_losses = []
    testing_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(train_data_input_batches_I_vec)):
            x_train_I = torch.cuda.FloatTensor(train_data_input_batches_I_vec[i])
            x_train_R = torch.cuda.FloatTensor(train_data_input_batches_R_vec[i])
            x_train_D = torch.cuda.FloatTensor(train_data_input_batches_D_vec[i])
            x_train_M = torch.cuda.FloatTensor(train_data_input_batches_M_vec[i])
            x_train_Y = torch.cuda.FloatTensor(train_data_input_batches_Y_vec[i])
            x_train_T = torch.cuda.FloatTensor(train_data_input_batches_T_vec[i])
            x_train_DW = torch.cuda.FloatTensor(train_data_input_batches_DW_vec[i])

            y_train = torch.cuda.FloatTensor(train_data_output_batches[i])
            optimizer.zero_grad()    
            # Forward pass
            y_pred = model(x_train_I, x_train_R, x_train_D, x_train_M, x_train_Y, x_train_T, x_train_DW)    
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_train)
            # Loss on original data for printing.
            y_orig_train = y_train * (Maxs[6] - Mins[6]) + Mins[6]
            y_orig_pred = y_pred.squeeze() * (Maxs[6] - Mins[6]) + Mins[6]
            orig_loss = torch.sqrt(criterion(y_orig_pred, y_orig_train))
            epoch_loss += orig_loss.data.item()

            # Backward pass
            loss.backward()
            optimizer.step()
        # Compute Loss over test data. 
        test_total_loss = 0
        test_total_num = 0
        for j in range(len(test_data_input_batches_I_vec)):
            x_test_I = torch.cuda.FloatTensor(test_data_input_batches_I_vec[j])
            x_test_R = torch.cuda.FloatTensor(test_data_input_batches_R_vec[j])
            x_test_D = torch.cuda.FloatTensor(test_data_input_batches_D_vec[j])
            x_test_M = torch.cuda.FloatTensor(test_data_input_batches_M_vec[j])
            x_test_Y = torch.cuda.FloatTensor(test_data_input_batches_Y_vec[j])
            x_test_T = torch.cuda.FloatTensor(test_data_input_batches_T_vec[j])
            x_test_DW = torch.cuda.FloatTensor(test_data_input_batches_DW_vec[j])

            y_test = torch.cuda.FloatTensor(test_data_output_batches[j])
            y_pred = model(x_test_I, x_test_R, x_test_D, x_test_M, x_test_Y, x_test_T, x_test_DW)

            y_orig_test = y_test * (Maxs[6] - Mins[6]) + Mins[6]
            y_orig_pred = y_pred.squeeze() * (Maxs[6] - Mins[6]) + Mins[6]
            # SQRT for RMSE
            test_loss = (y_orig_pred - y_orig_test)**2
            test_loss = torch.sum(test_loss)
            test_total_loss += test_loss.data.item()
            test_total_num += len(y_pred)
        # Append losses. 
        training_losses.append(epoch_loss/ len(train_data_input_batches_I_vec))
        testing_losses.append(test_total_loss / len(test_data_input_batches_I_vec))
        print("Epoch:", epoch, "Train Loss = ", epoch_loss/len(train_data_input_batches_I_vec), "Test Loss = " , math.sqrt(test_total_loss / test_total_num)) 
        

    # Preprocess Test data.
    testing_batch_I_vec = []
    testing_batch_R_vec = []
    testing_batch_D_vec = []
    testing_batch_M_vec = []
    testing_batch_Y_vec = []
    testing_batch_T_vec = []
    testing_batch_DW_vec = []

    for i in range(1, len(testing_data)+1):
        I, R, D, M, Y, TRX_T = testing_data[i-1]

        # Making one hot vectors. 
        I_vec = [0 for j in range(len(IDs))]
        I_vec[IDs.index(I)] = 1
        R_vec = [0 for j in range(len(REGs))]
        R_vec[REGs.index(R)] = 1
        D_vec = [0 for j in range(31)]
        D_vec[int(D)-1] = 1
        M_vec = [0 for j in range(12)]
        M_vec[int(M)-1] = 1
        Y_vec = [0, 0]
        if Y == 2019:
            Y_vec[1] = 1
        else:
            Y_vec[0] = 1
        T_vec = [0,0]
        T_vec[int(TRX_T)-1] = 1

        DW_vec = [0 for j in range(7)]
        d = datetime.datetime(int(Y), int(M), int(D))
        DW_vec[d.weekday()] = 1

        testing_batch_I_vec.append(I_vec)
        testing_batch_R_vec.append(R_vec)   
        testing_batch_D_vec.append(D_vec)
        testing_batch_M_vec.append(M_vec)
        testing_batch_Y_vec.append(Y_vec)
        testing_batch_T_vec.append(T_vec)
        testing_batch_DW_vec.append(DW_vec)

    # Predict values for Testing dataset.
    x_testing_I = torch.cuda.FloatTensor(testing_batch_I_vec)
    x_testing_R = torch.cuda.FloatTensor(testing_batch_R_vec)
    x_testing_D = torch.cuda.FloatTensor(testing_batch_D_vec)
    x_testing_M = torch.cuda.FloatTensor(testing_batch_M_vec)
    x_testing_Y = torch.cuda.FloatTensor(testing_batch_Y_vec)
    x_testing_T = torch.cuda.FloatTensor(testing_batch_T_vec)
    x_testing_DW= torch.cuda.FloatTensor(testing_batch_DW_vec)
    y_pred = model(x_testing_I, x_testing_R, x_testing_D, x_testing_M, x_testing_Y, x_testing_T, x_testing_DW)
    # Reverse normalization.
    Y_values = y_pred * (Maxs[6] - Mins[6]) + Mins[6]
    
    
    # Converting the Pytorch Tensor into numpy.
    Y_values = Y_values.cpu().detach().numpy()
    #print(Y_values)
    # Save values into csv file.
    np.savetxt("test_predictions.csv", Y_values, delimiter=",")

if __name__ == '__main__':
    main()
