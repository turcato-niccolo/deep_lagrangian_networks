import dill as pickle
import numpy as np
import torch

def init_env(args):

    # Set the NumPy Formatter:
    np.set_printoptions(suppress=True, precision=2, linewidth=500,
                        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model = bool(args.r[0]), bool(args.l[0]), bool(args.m[0])

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model


def load_dataset(n_characters=3, filename="data/DeLaN_Data.pickle"):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    n_dof = 2

    dt = np.concatenate([t[1:] - t[:-1] for t in data["t"]])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_char]

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean

def load_dataset_panda(filename_train="data/FE_panda/FE_panda_pybul_tr.pkl", filename_test="data/FE_panda/FE_panda_pybul_test.pkl"):

    data_train = pickle.load(open(filename_train, 'rb'))
    data_test = pickle.load(open(filename_test, 'rb'))

    n_dof = 7
    qp_labels = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'q_7']
    qv_labels = ['dq_1', 'dq_2', 'dq_3', 'dq_4', 'dq_5', 'dq_6', 'dq_7']
    qa_labels = ['ddq_1', 'ddq_2', 'ddq_3', 'ddq_4', 'ddq_5', 'ddq_6', 'ddq_7']
    tau_labels = ['tau_1', 'tau_2', 'tau_3', 'tau_4', 'tau_5', 'tau_6', 'tau_7']
    train_qp, train_qv, train_qa, train_tau = data_train[qp_labels].to_numpy(), data_train[qv_labels].to_numpy(), data_train[qa_labels].to_numpy(), data_train[tau_labels].to_numpy()
    test_qp, test_qv, test_qa, test_tau = data_test[qp_labels].to_numpy(), data_test[qv_labels].to_numpy(), data_test[qa_labels].to_numpy(), data_test[tau_labels].to_numpy()
    return (train_qp, train_qv, train_qa, train_tau), \
           (test_qp, test_qv, test_qa, test_tau)

    # divider = [0, ]   # Contains idx between characters for plotting

    # for i in range(len(data["labels"])):

    #     if i in test_idx:
    #         test_labels.append(data["labels"][i])
    #         test_qp = np.vstack((test_qp, data["qp"][i]))
    #         test_qv = np.vstack((test_qv, data["qv"][i]))
    #         test_qa = np.vstack((test_qa, data["qa"][i]))
    #         test_tau = np.vstack((test_tau, data["tau"][i]))

    #         test_m = np.vstack((test_m, data["m"][i]))
    #         test_c = np.vstack((test_c, data["c"][i]))
    #         test_g = np.vstack((test_g, data["g"][i]))

    #         test_p = np.vstack((test_p, data["p"][i]))
    #         test_pd = np.vstack((test_pd, data["pdot"][i]))
    #         divider.append(test_qp.shape[0])

    #     else:
    #         train_labels.append(data["labels"][i])
    #         train_qp = np.vstack((train_qp, data["qp"][i]))
    #         train_qv = np.vstack((train_qv, data["qv"][i]))
    #         train_qa = np.vstack((train_qa, data["qa"][i]))
    #         train_tau = np.vstack((train_tau, data["tau"][i]))

    #         train_p = np.vstack((train_p, data["p"][i]))
    #         train_pd = np.vstack((train_pd, data["pdot"][i]))

    # return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
    #        (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
    #        divider, dt_mean