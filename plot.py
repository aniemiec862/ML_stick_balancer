import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file]
        return data

def calculate_moving_average(data, window_size):
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return moving_avg

def calculate_standard_deviation(data, window_size):
    std_dev = np.std([data[i:i+window_size] for i in range(len(data)-window_size+1)], axis=1)
    return std_dev

def draw_moving_average(filename, window_size):
    data = read_data(filename)
    moving_avg = calculate_moving_average(data, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg, label='Moving Average', linewidth=2)
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Moving Average Plot')
    plt.grid(True)
    plt.show()

def draw_standard_deviation(file_names, window_size):
    data_frames = [pd.DataFrame({'Value': read_data(file_name)}) for file_name in file_names]
    averages = [df['Value'].rolling(window=window_size).mean() for df in data_frames]
    std_dev = [df['Value'].rolling(window=window_size).std() for df in data_frames]

    overall_average = sum(averages) / len(averages)
    overall_std_dev = sum(std_dev) / len(std_dev)

    plt.figure(figsize=(10, 6))
    plt.plot(overall_average, label='avg', color='blue')
    plt.plot(overall_average + overall_std_dev, label='avg + std_dev', linestyle='--', color='green')
    plt.plot(overall_average - overall_std_dev, label='avg - std_dev', linestyle='--', color='red')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Standard Deviation Plot')
    plt.grid(True)
    plt.show()


window_size = 10

draw_moving_average("result1.txt", window_size)

file_names = ["result1.txt", "result2.txt", "result3.txt"]
# draw_standard_deviation(file_names, window_size)
