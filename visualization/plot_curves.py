import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm

def load_data(data_path,algorithms):
    datasets = []
    for algorithm in algorithms:
        dataset = []
        data_path = os.path.join(data_path,algorithm)
        for data_name in os.listdir(data_path):
            data = np.load(os.path.join(data_path,data_name))
            dataset.append(np.array(data))
        datasets.append(dataset)
    return datasets

def get_statistics(dataset):
    dataset = np.array(dataset)
    mean = np.mean(dataset,axis=0)
    std = np.std(dataset, axis=0)
    return mean[0],std[0]

def learning_curve(args,algorithms):
    plt.style.use('ggplot')
    font_S = fm.FontProperties(family='Times New Roman', size=18, stretch=0)
    data_path = os.path.join(args.data_path,args.env_name)
    learning_datas = load_data(data_path, algorithms)
    plt.figure(figsize=(10, 6))
    for i in range(len(learning_datas)):
        average_reward, std_reward = get_statistics(learning_datas[i])
        episodes = range(len(average_reward))
        plt.plot(episodes, average_reward, linestyle='-',label=algorithms[i])
        plt.fill_between(episodes, average_reward - std_reward, average_reward + std_reward, alpha=0.3)
    plt.title(args.env_name, fontproperties=font_S)
    plt.xlabel('Timestep', fontproperties=font_S)
    plt.ylabel('Average Reward', fontproperties=font_S)
    plt.legend(fontsize=18, prop = font_S, loc='lower right')
    plt.xticks(fontproperties=font_S)
    plt.yticks(fontproperties=font_S)
    plt.grid(True)
    plt.savefig(os.path.join(args.curve_path,args.env_name+'.pdf'),dpi=350)