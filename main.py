import csv
import matplotlib.pyplot as plt
import math


class Point:
    def __init__(self, x1: float, x2: float, label: int):
        self.x1 = x1
        self.x2 = x2
        self.label = label


def S(w1, x1, w2, x2, b):
    return 1 / (1 + math.e ** -(w1 * x1 + w2 * x2 + b))


def dcost_dy(label, y):
    return -(label * (1 / (y * math.log(10, math.e))) + (label - 1) * (
            1 / ((1 - y) * math.log(10, math.e))))


def dcost_dzu(y, label, z0, u0, z1, u1, b2):
    return 2 * (y - label) * S(z0, u0, z1, u1, b2) * (1 - S(z0, u0, z1, u1, b2))


def train1(train_data):
    w1, w2, b = 0.5, 0.5, 0.5
    n_epoch = 2000
    lr = 0.1
    n = 140
    cost = 0
    for i in range(n_epoch):
        grad_w = [0, 0]
        grad_b = 0
        for j in train_data:
            y = S(w1, j.x1, w2, j.x2, b)
            cost = -(j.label * math.log(y, 2) + (1 - j.label) * math.log(1 - y, 2))

            grad_w[0] += dcost_dy(j.label, y) * S(w1, j.x1, w2, j.x2, b) * (1 - S(w1, j.x1, w2, j.x2, b)) * j.x1
            grad_w[1] += dcost_dy(j.label, y) * S(w1, j.x1, w2, j.x2, b) * (1 - S(w1, j.x1, w2, j.x2, b)) * j.x2
            grad_b += dcost_dy(j.label, y) * S(w1, j.x1, w2, j.x2, b) * (1 - S(w1, j.x1, w2, j.x2, b))
        # print("grad_w", grad_w)
        w1 = w1 - (lr * grad_w[0]) / n
        w2 = w2 - (lr * grad_w[1]) / n
        b = b - (lr * grad_b) / n

    return w1, w2, b


def train2(train_data):
    w1, w2, b0 = 0.5, 0.5, 0.5
    z0, z1, u0, u1, v0, v1 = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    b1, b2 = 0.5, 0.5
    n_epoch = 3000
    lr = 0.05
    n = 140
    for i in range(n_epoch):
        # grad_w = [0, 0]
        # grad_v = [0, 0]
        # grad_u = [0, 0]
        # grad_b0 = 0
        # grad_b1 = 0
        # grad_b2 = 0
        for j in train_data:
            z0 = S(w1, j.x1, w2, j.x2, b0)
            z1 = S(v0, j.x1, v1, j.x2, b1)
            y = S(u0, z0, u1, z1, b2)

            temp = dcost_dzu(y, j.label, z0, u0, z1, u1, b2) * u0 * S(j.x1, w1, j.x2, w2, b0) \
                     * (1 - S(j.x1, w1, j.x2, w2, b0))
            grad_w1 = temp * j.x1
            grad_w2 = temp * j.x2
            grad_b0 = temp

            # grad_w[1] = dcost_dy(j.label, y) * S(w1, j.x1, w2, j.x2, b) * (1 - S(w1, j.x1, w2, j.x2, b)) * j.x2

            temp = dcost_dzu(y, j.label, z0, u0, z1, u1, b2) * u1 * S(j.x1, v0, j.x2, v1, b1) \
                     * (1 - S(j.x1, v0, j.x2, v1, b1))
            grad_v0 = temp * j.x1
            grad_v1 = temp * j.x2
            grad_b1 = temp

            temp = dcost_dzu(y, j.label, z0, u0, z1, u1, b2)
            grad_u0 = temp * z0
            grad_u1 = temp * z1
            grad_b2 = temp

            w1 = w1 - (lr * grad_w1)
            w2 = w2 - (lr * grad_w2)
            u0 = u0 - (lr * grad_u0)
            u1 = u1 - (lr * grad_u1)
            v0 = v0 - (lr * grad_v0)
            v1 = v1 - (lr * grad_v1)
            b0 = b0 - (lr * grad_b0)
            b1 = b1 - (lr * grad_b1)
            b2 = b2 - (lr * grad_b2)
        # print("grad_w", grad_w)

    return w1, w2, b0, v0, v1, b1, u0, u1, b2


if __name__ == "__main__":
    points_0 = []
    points_1 = []
    points = []
    test_x1 = []
    test_x2 = []
    test_label = []
    test_points = []
    with open('dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if not line_count == 0 and line_count > 40:
                if row[2] == '0':
                    points_0.append(Point(float(row[0]), float(row[1]), int(row[2])))
                else:
                    points_1.append(Point(float(row[0]), float(row[1]), int(row[2])))
                points.append(Point(float(row[0]), float(row[1]), int(row[2])))
            elif not line_count == 0 and line_count < 41:
                test_x1.append(float(row[0]))
                test_x2.append(float(row[1]))
                test_label.append(float(row[2]))
                test_points.append(Point(float(row[0]), float(row[1]), int(row[2])))
            line_count += 1

        # create plot

        x_array0, y_array0 = [], []
        for point in points_0:
            x_array0.append(point.x1)
            y_array0.append(point.x2)
        plt.scatter(x_array0, y_array0, marker='*', color='red')

        x_array1, y_array1 = [], []
        for point in points_1:
            x_array1.append(point.x1)
            y_array1.append(point.x2)
        plt.scatter(x_array1, y_array1, marker='*', color='green')

        # w1, w2, b = train1(points)
        # print(w1, w2, b)

        w0, w1, b0, v0, v1, b1, u0, u1, b2 = train2(points)



        accuracy = 0
        x1_test0 = []
        x2_test0 = []
        x1_test1 = []
        x2_test1 = []
        for i in test_points:
            # y = S(w1, i.x1, w2, i.x2, b)
            z0 = S(i.x1, w0, i.x2, w1, b0)
            z1 = S(i.x1, v0, i.x2, v1, b1)
            y = S(z0, u0, z1, u1, b2)
            if (y > 0.5 and i.label == 1) or (y < 0.5 and i.label == 0):
                accuracy += 1
            if y > 0.5:
                x2_test1.append(i.x2)
                x1_test1.append(i.x1)
            else:
                x2_test0.append(i.x2)
                x1_test0.append(i.x1)

        plt.scatter(x1_test0, x2_test0, marker='*', color='yellow')
        plt.scatter(x1_test1, x2_test1, marker='*', color='blue')

        accuracy /= 40
        print("accuracy= ", accuracy)

        plt.xlabel('x1', fontsize=16)
        plt.ylabel('x2', fontsize=16)
        plt.title('first neural network', fontsize=20)
        plt.show()
