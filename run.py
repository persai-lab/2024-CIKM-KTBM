import torch
from easydict import EasyDict
import pickle
from trainer.trainer import trainer
import matplotlib.pyplot as plt
import numpy as np

def circle_points(r, n):
    """
    generate evenly distributed unit divide vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


def paretoMLT_exp(config):
    config = EasyDict(config)

    ref_vec = torch.tensor(circle_points([1], [config.num_pref])[0]).float()

    data = pickle.load(open('data/{}/train_val_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']
    config.num_nongradable_items = data['num_items_L']
    config.num_users = data['num_users']

    print(config)

    # for i in range(config.num_pref):
    for i in range(2,3): #3th is the middle preference vector
        pref_idx = i
        exp_trainner = trainer(config, data, ref_vec, pref_idx)
        exp_trainner.train()



def ednet():
    config = {
                "data_name": 'ednet',
                "model_name": "KTBM",

                "mode": 'test',
                "fold": 1,
                "metric": 'auc',
                "shuffle": True,

                "cuda": True,
                "gpu_device": 0,
                "seed": 1024,

                "n_tasks": 2,
                "num_pref": 5,  # number of dividing vector

                "min_seq_len": 2,
                "max_seq_len": 100,  # the max step of RNN model
                "batch_size": 32,
                "learning_rate": 0.01,
                "max_epoch": 70,
                "validation_split": 0.2,

                "embedding_size_q": 64,
                "embedding_size_a": 32,
                "embedding_size_l": 32,
                "embedding_size_d": 16,
                "embedding_size_q_behavior": 16,
                "embedding_size_l_behavior": 16,
                "num_concepts": 8,
                "key_dim": 32,
                "value_dim": 32,
                "summary_dim": 32,
                "behavior_map_size": 32,
                "behavior_hidden_size": 32,
                "behavior_summary_fc": 32,

                'weight_type': 0.1,

                "init_std": 0.2,
                "max_grad_norm": 10,

                "optimizer": 'adam',
                "epsilon": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.05,
            }
    paretoMLT_exp(config)


def Junyi2063():
    config = {
                "data_name": 'Junyi2063',
                "model_name": 'KTBM',
                # "model_name": 'MANN',

                "mode": 'test',
                "fold": 1,
                "metric": 'auc',
                "shuffle": True,

                "n_tasks": 2,
                "num_pref": 5,  # number of dividing vector

                "cuda": True,
                "gpu_device": 0,
                "seed": 1024,

                "min_seq_len": 2,
                "max_seq_len": 100,  # the max seq len of model
                "batch_size": 32,
                "learning_rate": 0.01,
                "max_epoch": 60,
                "validation_split": 0.2,

                "embedding_size_q": 32,
                "embedding_size_a": 32,
                "embedding_size_l": 32,
                "embedding_size_d": 16,
                "embedding_size_q_behavior": 32,
                "embedding_size_l_behavior": 32,
                "num_concepts": 32,
                "key_dim": 64,
                "value_dim": 64,
                "summary_dim": 32,
                "behavior_map_size": 32,
                "behavior_hidden_size": 64,
                "behavior_summary_fc":32,

                'weight_type': 0.15,

                "init_std": 0.2,
                "max_grad_norm": 50,

                "optimizer": 'adam',
                "epsilon": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.05,
            }
    paretoMLT_exp(config)


def morf():
    config = {
        "data_name": 'MORF686',
        "model_name": 'KTBM',

        "mode": 'test',
        "fold": 3,
        "metric": 'rmse',
        "shuffle": True,

        "n_tasks": 2,
        "num_pref": 5,  # number of dividing vector

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "min_seq_len": 2,
        "max_seq_len": 100,  # the max step of RNN model
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_epoch": 200,
        "validation_split": 0.2,

        "embedding_size_q": 16,
        "embedding_size_a": 16,
        "embedding_size_l": 16,
        "embedding_size_d": 8,
        "embedding_size_q_behavior": 8,
        "embedding_size_l_behavior": 8,
        "num_concepts": 8,
        "key_dim": 16,
        "value_dim": 16,
        "summary_dim": 16,
        "behavior_map_size": 16,
        "behavior_hidden_size": 32,
        "behavior_summary_fc": 16,

        'weight_type': 0.1,

        "init_std": 0.2,
        "max_grad_norm": 10,

        "optimizer": 'adam',
        "epsilon": 0.1,
        # "epsilon": 1e-8,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
    }
    paretoMLT_exp(config)



if __name__== '__main__':
    ednet()
    # Junyi2063()
    # morf()