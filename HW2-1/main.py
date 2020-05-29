import csv
import pandas as pd
import os
import random
import seaborn as sb
import matplotlib.pyplot as plt
import pygal.maps.world
from pygal.maps.world import COUNTRIES
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn

from Dataset import torch_dataset
from LSTM import LSTMModel
from GRU import GRUModel
from Trainer import Trainer

file = 'covid_19.csv'

threshold = 0.8
INTERVAL = 3

BATCH_SIZE = 128
HIDDEN_DIM = 10
INPUT_SIZE = 1
OUTPUT_SIZE = 2
EPOCH = 30

def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code
    # If the country wasn't found, return None.
    return None

def make_label(confirmed_days, countrys):
    # dataset = {}
    dataset = []
    for country in countrys:
        # print(country)
        sequence = []
        for day in range(0, len(confirmed_days[country]) - INTERVAL):  # 0~76
            subsequence = confirmed_days[country][day:day + INTERVAL].tolist()  # data of L days
            label = confirmed_days[country][day + INTERVAL] > confirmed_days[country][day + INTERVAL - 1]
            # L+1 day
            dataset.append([subsequence, label])
        # dataset.append(sequence)
    # print(dataset)
    return dataset


if __name__ == '__main__':
    with open(file, newline='') as csvfile:
        rows = csv.reader(csvfile)
        list_of_rows = list(rows)
        # Country_index = range(3, 13) # for print the correlation coefficients
        Country_index = range(3, len(list_of_rows))
        Day = range(3, len(list_of_rows[0]))

    # computer difference sequence
    Country = []
    corr_df = pd.DataFrame()
    for c in Country_index:
        # column name
        country = list_of_rows[c][0]
        # sum of people
        accumulation = list_of_rows[c][3:len(list_of_rows[0])]
        person_per_day = []
        Country.append(country)
        for i in range(len(accumulation)):
            if i == 0:
                person_per_day.append(int(accumulation[i]) - 0)
            else:
                t = int(accumulation[i]) - int(accumulation[i - 1])
                if t < 0: t = 0
                person_per_day.append(t)
        # build the column
        corr_df[country] = person_per_day

    print(corr_df)
    correlation = corr_df.corr(method='pearson')
    print(correlation)
    # collect countries in a set denoted as C.
    selected_countrys = []
    for index, row in correlation.iterrows():
        for row_c in row[row > threshold].index:
            if row_c != index and row_c not in selected_countrys:
                selected_countrys.append(row_c)

    print(len(selected_countrys), selected_countrys)  # selected countrys (C)
    ##### plot the correlation coefficients #####

    figure = sb.heatmap(correlation).get_figure()
    figure.tight_layout()
    figure.savefig('correlation coefficient.png')
    #############################################

    data = make_label(corr_df, selected_countrys)
    # print(data)
    random.shuffle(data)
    # key : country [C]
    # subsequence index [0]~[77]
    # subsequence, label [0], [1]
    # print(len(data))
    TrainData = torch_dataset(data)
    # Split the whole training data into train and test
    pivot = len(TrainData) * 7 // 10
    print('The number of data:', len(TrainData))

    train_set = Subset(TrainData, range(0, pivot))
    test_set = Subset(TrainData, range(pivot, len(TrainData)))
    # Make the corresponding dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)

    my_lstm = LSTMModel(INPUT_SIZE, HIDDEN_DIM, OUTPUT_SIZE)
    print(my_lstm)

    my_gru = GRUModel(INPUT_SIZE, HIDDEN_DIM, OUTPUT_SIZE)
    print(my_gru)

    os.makedirs('./log', exist_ok=True)
    print('LSTM training starts...')
    Trainer(interval=INTERVAL, dataloader=(train_loader, test_loader), net=my_lstm, num_epochs=EPOCH).run()
    print('GRU training starts...')
    Trainer(interval=INTERVAL, dataloader=(train_loader, test_loader), net=my_gru, num_epochs=EPOCH).run()

    # 要預測的最後三天
    pred_df = corr_df[-INTERVAL:]
    ascending_dict = { }
    descending_dict = { }
    pth = 'LSTMModel_L3_30epochs.pth'
    device = torch.device('cuda')
    pred_NET = LSTMModel(INPUT_SIZE, HIDDEN_DIM, OUTPUT_SIZE).to(device)
    pred_NET.load_state_dict(torch.load(pth, map_location=device))
    pred_NET.eval()

    print('Compute the probability for each country...')
    for c in range(len(pred_df.columns)):
        country = pred_df.columns[c]
        country = get_country_code(country)
        if country:
            sequence = pred_df[pred_df.columns[c]].values

            sequence = torch.FloatTensor(sequence)
            sequence = sequence.view(-1, INTERVAL, 1)
            output = pred_NET(sequence.to(device)).to('cpu')
            softmax = nn.Softmax(dim=1)
            prob = softmax(output)
            prob, state = torch.max(prob, 1)
            if state == 1:
                ascending_dict[country] = int(prob.detach().numpy()[0]*100)
            else:
                descending_dict[country] = int(prob.detach().numpy()[0]*100)

    worldmap_chart = pygal.maps.world.World()
    worldmap_chart.add('ascending', ascending_dict)
    worldmap_chart.add('descending', descending_dict)
    # print('ascending_dict', ascending_dict)
    # print('descending_dict', descending_dict)
    worldmap_chart.render_to_file('chart.svg')
