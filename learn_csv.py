import csv
import os

csv_path = "csv/"
FILENAME = 'hand_data.csv'


def make_learningCsv(filename: str, size: int, angle: int, num: list):
    with open(csv_path+FILENAME, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)

        csvfile = open(csv_path + filename + ".csv", "w")
        w = csv.writer(csvfile)

        label_list = [0, 0, 0, 0, 0, 0]

        for i, row in enumerate(csvreader):
            if row == []:
                continue

            if i == 0:
                shape = row
                high = int(shape[0])
                wide = int(shape[1])
                continue

            hand_num = int(row[0])
            del(row[0])
            csv_size = int(row[0])
            del(row[0])
            csv_angle = int(row[0])
            del(row[0])
            if size < csv_size or angle < csv_angle:
                continue

            if label_list[hand_num] >= num[hand_num]:
                continue

            csv_list = []
            csv_list.append(hand_num)

            landmark_list = []
            for x, y in zip(row[0::2], row[1::2]):
                landmark_list.append([float(x), float(y)])

            csv_list = []
            csv_list.append(hand_num)
            for i in landmark_list:
                x_point = int(int(wide/csv_size)*i[0])
                y_point = int(int(high/csv_size)*i[1])
                csv_list.append(x_point)
                csv_list.append(y_point)
            w.writerow(csv_list)
            label_list[hand_num] += 1

        csvfile.close()
    print(
        f"0:{label_list[0]}, 1:{label_list[1]}, 2:{label_list[2]}, 3:{label_list[3]}, 4:{label_list[4]}, 5:{label_list[5]}")
    print("終了しました")


file_name = input("csvファイル名の入力")
if os.path.exists(csv_path + file_name + ".csv") is True:
    print("ファイルが存在します")
    print("終了")
size = int(input("画像サイズの指定（1~4）"))
rot = int(input("角度の指定(5の倍数 0~90)"))
print("取得するデータ数の指定")
os.path.exists('hello.txt')
num_list = []
for i in range(6):
    num_list[i] = int(input(f"{i}　のデータ数の指定"))
one = int(input(""))
make_learningCsv(file_name, size, rot, num_list)
# make_learningCsv("hand0_8", 3, 30, [4000, 5000, 5000, 5000, 3500, 5000])
