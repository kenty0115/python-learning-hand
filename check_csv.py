import csv

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

csv_path = "csv/"
filename = input("ファイル名の指定")

with open(csv_path + filename, encoding='utf8', newline='') as f:
    csvreader = csv.reader(f)

    num_list = [0, 0, 0, 0, 0, 0]
    first = True
    for row in csvreader:

        if row == []:
            continue

        label = int(row[0])
        if label == 0:
            num_list[0] = num_list[0] + 1
        elif label == 1:
            num_list[1] = num_list[1] + 1
        elif label == 2:
            num_list[2] = num_list[2] + 1
        elif label == 3:
            num_list[3] = num_list[3] + 1
        elif label == 4:
            num_list[4] = num_list[4] + 1
        else:
            num_list[5] = num_list[5] + 1
    print(num_list)
print("終了")
