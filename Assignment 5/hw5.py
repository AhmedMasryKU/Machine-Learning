from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy


D = 1

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

def learn(node_indices, is_terminal, need_split, node_values, node_features, node_splits, P, data_train_waiting, data_train_eruptions):
    node_indices, is_terminal, need_split, node_values, node_features, node_splits = copy.deepcopy(node_indices), copy.deepcopy(is_terminal), copy.deepcopy(need_split), copy.deepcopy(node_values), copy.deepcopy(node_features), copy.deepcopy(node_splits)
    while True:
        split_nodes = np.argwhere(need_split==True)
        if len(split_nodes) == 0:
            break
        split_nodes = split_nodes.reshape(len(split_nodes),)
        for split_node in split_nodes:
            data_points_indices = node_indices[split_node]
            need_split[split_node] = False
            #for i in data_points_indices:
            #    print(i)
            node_values[split_node] = np.mean([data_train_waiting[i] for i in data_points_indices])
            if len(data_points_indices) <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                best_scores = np.zeros(D)
                best_splits = np.zeros(D)
                for d in range(D):
                    uni_vals = np.unique([data_train_eruptions[l] for l in data_points_indices])
                    split_poses = [(uni_vals[i] + uni_vals[i+1]) / 2 for i in range(len(uni_vals)-1)]
                    split_scores = np.zeros(len(split_poses))
                    for p in range(len(split_poses)):
                        #print("Data: ", np.argwhere([data_train_waiting[i] for i in data_points_indices] >= split_poses[p]))
                        left_inds = [data_points_indices[i[0]] for i in np.argwhere([data_train_eruptions[i] for i in data_points_indices] < split_poses[p])]
                        right_inds = [data_points_indices[i[0]] for i in np.argwhere([data_train_eruptions[i] for i in data_points_indices] >= split_poses[p])]
                        #print("Right inds: ", right_inds)
                        #print("Left inds: ", left_inds)
                        left_mean = np.mean([data_train_waiting[i] for i in left_inds])
                        right_mean = np.mean([data_train_waiting[i] for i in right_inds])
                        split_scores[p] = - (1/len(data_points_indices)) * ( np.sum([(k-left_mean)**2 for k in [data_train_waiting[i] for i in left_inds]]) + np.sum([(k-right_mean)**2 for k in [data_train_waiting[i] for i in right_inds]]))
                    best_scores[d] = np.max(split_scores)
                    best_splits[d] = split_poses[np.argmax(split_scores)]
                split_feat = np.argmax(best_scores)
                node_features[split_node] = split_feat
                node_splits[split_node] = best_splits[split_feat]

                left_inds = np.array([data_points_indices[i][0] for i in np.argwhere([data_train_eruptions[k] for k in data_points_indices]< best_splits[split_feat])])
                node_indices[2*split_node+1] = left_inds
                is_terminal[2*split_node+1] = False
                need_split[2*split_node+1] = True

                right_inds = np.array([data_points_indices[i][0] for i in np.argwhere([data_train_eruptions[k] for k in data_points_indices]>= best_splits[split_feat])])
                node_indices[2*split_node + 2] = right_inds
                is_terminal[2*split_node+2] = False
                need_split[2*split_node+2] = True
    return node_indices, is_terminal, need_split, node_values, node_features, node_splits


def plot(borders, y_values, data_train_eruptions, data_train_waiting, data_test_eruptions, data_test_waiting, P):
    plt.scatter(data_train_eruptions, data_train_waiting, label="Train")
    plt.scatter(data_test_eruptions, data_test_waiting, label="Test")

    # borders= np.append(left_borders, right_borders[len(right_borders)-1])
    # hist = np.append(hist_out, hist_out[len(hist_out)-1])
    plt.step(borders, y_values, where='post', color='black')
    plt.ylabel('Waiting time to next eruption (min)')
    plt.xlabel('Eruption time (min)')
    plt.title("P = " + str(P))
    plt.legend()
    plt.show()

def predict(x_values, is_terminal_l, node_values_l, node_splits_l):
    y_values = np.zeros(len(x_values))
    for i in range(len(x_values)):
        ind = 0
        while True:
            if is_terminal_l[ind]:
                y_values[i] = node_values_l[ind]
                break
            else:
                if x_values[i] <= node_splits_l[ind]:
                    ind = ind*2+1
                else:
                    ind = ind*2+2
    return y_values

def RMSE(y_test, y_predict):
    return np.sqrt(np.mean([(y_test[i] - y_predict[i])**2 for i in range(len(y_test))]))

def plot_rmses(values):
    p = [i[0] for i in values]
    rmses = [i[1] for i in values]
    plt.plot(p, rmses, marker='o', color='black')
    plt.ylabel('RMSE')
    plt.xlabel('Pre-pruning size (p)')
    plt.show()

def main():
    # Load data. 
    data = np.array(read_csv("hw05_data_set.csv"))

    # Split the data. 
    data_train = data[0:150]
    data_test = data[150:272]

    data_train_eruptions = [i[0] for i in data_train]
    data_train_waiting = [i[1] for i in data_train]

    data_test_eruptions = [i[0] for i in data_test]
    data_test_waiting = [i[1] for i in data_test]

    data_eruptions = [i[0] for i in data]
    data_waiting = [i[1] for i in data]

    N = len(data)
    N_training = len(data_train)
    N_testing = len(data_test)
    P = 25

    max_node_num = 20000
    node_indices = [[]] * max_node_num
    node_indices[0] = np.array([i for i in range(N_training)])
    is_terminal = np.full(max_node_num, False)
    need_split = np.full(max_node_num, False)
    need_split[0] = True
    node_values = np.zeros(max_node_num)
    node_features = np.zeros(max_node_num)
    node_splits = np.zeros(max_node_num)

    node_indices_l, is_terminal_l, need_split_l, node_values_l, node_features_l, node_splits_l = \
        learn(node_indices, is_terminal, need_split, node_values, node_features, node_splits, P, data_train_waiting, data_train_eruptions)

    x_borders = np.arange(np.min(data_eruptions)-0.1, np.max(data_eruptions)+0.1, 0.01)
    y_borders = predict(x_borders, is_terminal_l, node_values_l, node_splits_l)
    plot(x_borders, y_borders, data_train_eruptions, data_train_waiting, data_test_eruptions, data_test_waiting, P)

    rmse = RMSE(data_test_waiting, predict(data_test_eruptions, is_terminal_l, node_values_l, node_splits_l))
    print("RMSE is ", rmse, " when P is ", P)

    rmses = []
    for p in range(5, 55, 5):
        node_indices_l, is_terminal_l, need_split_l, node_values_l, node_features_l, node_splits_l = \
            learn(node_indices, is_terminal, need_split, node_values, node_features, node_splits, p, data_train_waiting, data_train_eruptions)
        rmse = RMSE(data_test_waiting, predict(data_test_eruptions, is_terminal_l, node_values_l, node_splits_l))
        rmses.append((p, rmse))

    plot_rmses(rmses)

if __name__=='__main__':
    main()