import numpy as np
from multihead_models import Vanilla_NN, MFVI_NN
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_vcl(hidden_size, no_epochs, data_gen, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    gans = []
    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=mf_weights)

        mf_model.train(x_train, y_train, head, no_epochs, bsize)

        mf_model.update_prior()
        # Save weights before test (and last-minute training on coreset
        mf_model.save_weights()

        acc = test

        mf_model.load_weights()
        mf_model.clean_copy_weights()

        if not single_head:
            mf_model.create_head()

    return all_acc

