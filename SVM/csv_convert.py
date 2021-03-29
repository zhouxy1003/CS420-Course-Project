import sys
import csv
import numpy as np


def empty_table(input_file):  # 建立空表格， 维数为原数据集中最大特征维数
    max_feature = 0
    count = 0
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=" ")
        for line in reader:
            count += 1
            for i in line:
                try:
                    num = int(i.split(":")[0])
                except ValueError:
                    pass
                if num > max_feature:
                    max_feature = num

    return np.zeros((count, max_feature + 1))


def write(input_file, output_file, table):
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=" ")
        for c, line in enumerate(reader):
            label = line.pop(0)
            table[c, 0] = label
            if line[-1].strip() == '':
                line.pop(-1)

            line = map(lambda x: tuple(x.split(":")), line)
            for i, v in line:
                i = int(i)
                table[c, i] = v

    # delete_col = []
    # for col in range(table.shape[1]):
    #     if not any(table[:, col]):
    #         delete_col.append(col)
    #
    # table = np.delete(table, delete_col, axis=1)  # 删除全 0 列
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in table:
            writer.writerow(line)


if __name__ == "__main__":
    input_file = 'gisette_scalet.txt'
    output_file = 'gisette_scalet.csv'
    table = empty_table(input_file)
    write(input_file, output_file, table)