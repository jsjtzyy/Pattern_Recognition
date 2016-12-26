# This code from scratch part is completed by ghe10, yingyiz2 and gjin7
import numpy as np
from random import shuffle


def readData(folder):
    x = []
    y = []
    target = []
    file = open("hw1/" + folder + "/lab/hw2" + folder +"_labels.txt")
    tmp_file = []

    for line in file:
        tmp_file.append(line)
    shuffle(tmp_file)
    file.close()

    i = 0
    for lines in tmp_file:
        tmp_y = np.zeros((1, 9))
        words = lines.split()
        num = int(words[0])
        tmp_y[0, num] = 1
        f = open("hw1/" + words[1])
        raw_data = f.read()
        f.close()
        tmp_x = []
        valid = 1
        count = 0
        for number in raw_data.split():
            if np.isnan(float(number)) or float(number) == np.inf or float(number) == -np.inf:
                valid = 0
                break
            tmp_x.append(float(number))
            count += 1
            if count == 70 * 16 or valid == 0:
                break
        if valid == 1 and count == 1120:
            target.append(num)
            x.append(np.array(tmp_x).reshape(1120, 1))
            i += 1
            y.append(np.transpose(tmp_y))
    print("read finished")
    print(i)
    return x, y, i, target

# if __name__ == '__main__':
#     readData("hw1/train/lab/hw2train_labels.txt")
