import torch
import numpy as np

def fixed_walk_wo_wall(model):
    # construct position and action
    action_one_hot_value_numpy = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.a_dim, model.hparams.gtm_sm.total_dim - 1), np.float32)
    position = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.s_dim, model.hparams.gtm_sm.total_dim), np.int32)
    action_selection = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.total_dim - 1), np.int32)
    for index_sample in range(model.hparams.train.batch_size):
        new_continue_action_flag = True
        L = 0
        order = [3, 0, 2, 1]
        for t in range(model.hparams.gtm_sm.total_dim):
            if t == 0:
                position[index_sample, :, t] = np.ones(2) * 4
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False
                    action_random_selection = order[L]
                    L = (L + 1) % 4
                    action_duriation = 20

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[index_sample, 1, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                                action_selection[index_sample, t - 1] = action_random_selection
                        elif action_random_selection == 1:
                            if position[index_sample, 1, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                                action_selection[index_sample, t - 1] = action_random_selection
                        elif action_random_selection == 2:
                            if position[index_sample, 0, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                                action_selection[index_sample, t - 1] = action_random_selection
                        elif action_random_selection == 3:
                            if position[index_sample, 0, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                                action_selection[index_sample, t - 1] = action_random_selection
                    else:
                        position[index_sample, :, t] = position[index_sample, :, t - 1]
                        action_selection[index_sample, t - 1] = 4
                    action_duriation -= 1
                else:
                    action_selection[index_sample, t - 1] = 4
                    position[index_sample, :, t] = position[index_sample, :, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True

    for index_sample in range(model.hparams.train.batch_size):
        action_one_hot_value_numpy[
            index_sample, action_selection[index_sample], np.array(range(model.hparams.gtm_sm.total_dim - 1))] = 1

    action_one_hot_value = torch.from_numpy(action_one_hot_value_numpy).to(device=model.device)

    return action_one_hot_value, position, action_selection    


def fixed_walk(model):
    # construct position and action
    action_one_hot_value_numpy = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.a_dim, model.hparams.gtm_sm.total_dim - 1), np.float32)
    position = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.s_dim, model.hparams.gtm_sm.total_dim), np.int32)
    action_selection = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.total_dim - 1), np.int32)
    for index_sample in range(model.hparams.train.batch_size):
        new_continue_action_flag = True
        L = 0
        order = [3, 0, 2, 1]
        for t in range(model.hparams.gtm_sm.total_dim):
            if t == 0:
                #position[index_sample, :, t] = np.random.randint(0, 9, size=(2))
                position[index_sample, :, t] = np.ones(2) * 4
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False
                    action_random_selection = order[L]
                    L = (L + 1) % 4
                    action_duriation = 20

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[index_sample, 1, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                        elif action_random_selection == 1:
                            if position[index_sample, 1, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                        elif action_random_selection == 2:
                            if position[index_sample, 0, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                        elif action_random_selection == 3:
                            if position[index_sample, 0, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                    else:
                        position[index_sample, :, t] = position[index_sample, :, t - 1]
                    action_duriation -= 1
                    action_selection[index_sample, t - 1] = action_random_selection
                else:
                    action_selection[index_sample, t - 1] = 4
                    position[index_sample, :, t] = position[index_sample, :, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True

    for index_sample in range(model.hparams.train.batch_size):
        action_one_hot_value_numpy[
            index_sample, action_selection[index_sample], np.array(range(model.hparams.gtm_sm.total_dim - 1))] = 1

    action_one_hot_value = torch.from_numpy(action_one_hot_value_numpy).to(device=model.device)

    return action_one_hot_value, position, action_selection


def random_walk_wo_wall(model):
    # construct position and action
    action_one_hot_value_numpy = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.a_dim, model.hparams.gtm_sm.total_dim - 1), np.float32)
    position = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.s_dim, model.hparams.gtm_sm.total_dim), np.int32)
    action_selection = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.total_dim - 1), np.int32)
    for index_sample in range(model.hparams.train.batch_size):
        new_continue_action_flag = True
        for t in range(model.hparams.gtm_sm.total_dim):
            if t == 0:
                #position[index_sample, :, t] = np.random.randint(0, 9, size=(2))
                position[index_sample, :, t] = np.ones(2) * 4
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False

                    while 1:
                        action_random_selection = np.random.randint(0, 4, size=(1))
                        if not (action_random_selection == 0 and position[index_sample, 1, t - 1] == 8):
                            if not (action_random_selection == 1 and position[index_sample, 1, t - 1] == 0):
                                if not (action_random_selection == 2 and position[index_sample, 0, t - 1] == 0):
                                    if not (action_random_selection == 3 and position[index_sample, 0, t - 1] == 8):
                                        break

                    #action_random_selection = np.random.randint(0, 5, size=(1))
                    action_duriation = np.random.poisson(2, 1)

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[index_sample, 1, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                                action_selection[index_sample, t - 1] = action_random_selection
                        elif action_random_selection == 1:
                            if position[index_sample, 1, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                                action_selection[index_sample, t - 1] = action_random_selection
                        elif action_random_selection == 2:
                            if position[index_sample, 0, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                                action_selection[index_sample, t - 1] = action_random_selection
                        elif action_random_selection == 3:
                            if position[index_sample, 0, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t - 1] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                                action_selection[index_sample, t - 1] = action_random_selection
                    else:
                        position[index_sample, :, t] = position[index_sample, :, t - 1]
                        action_selection[index_sample, t - 1] = 4
                    action_duriation -= 1
                else:
                    action_selection[index_sample, t - 1] = 4
                    position[index_sample, :, t] = position[index_sample, :, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True


    for index_sample in range(model.hparams.train.batch_size):
        action_one_hot_value_numpy[
            index_sample, action_selection[index_sample], np.array(range(model.hparams.gtm_sm.total_dim - 1))] = 1

    action_one_hot_value = torch.from_numpy(action_one_hot_value_numpy).to(device=model.device)

    return action_one_hot_value, position, action_selection


def random_walk(model):
    # construct position and action
    action_one_hot_value_numpy = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.a_dim, model.hparams.gtm_sm.total_dim - 1), np.float32)
    position = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.s_dim, model.hparams.gtm_sm.total_dim), np.int32)
    action_selection = np.zeros((model.hparams.train.batch_size, model.hparams.gtm_sm.total_dim - 1), np.int32)
    for index_sample in range(model.hparams.train.batch_size):
        new_continue_action_flag = True
        for t in range(model.hparams.gtm_sm.total_dim):
            if t == 0:
                #position[index_sample, :, t] = np.random.randint(0, 9, size=(2))
                position[index_sample, :, t] = np.ones(2) * 4
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False

                    while 1:
                        action_random_selection = np.random.randint(0, 4, size=(1))
                        if not (action_random_selection == 0 and position[index_sample, 1, t - 1] == 8):
                            if not (action_random_selection == 1 and position[index_sample, 1, t - 1] == 0):
                                if not (action_random_selection == 2 and position[index_sample, 0, t - 1] == 0):
                                    if not (action_random_selection == 3 and position[index_sample, 0, t - 1] == 8):
                                        break

                    #action_random_selection = np.random.randint(0, 5, size=(1))
                    action_duriation = np.random.poisson(2, 1)

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[index_sample, 1, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                        elif action_random_selection == 1:
                            if position[index_sample, 1, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                        elif action_random_selection == 2:
                            if position[index_sample, 0, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                        elif action_random_selection == 3:
                            if position[index_sample, 0, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                    else:
                        position[index_sample, :, t] = position[index_sample, :, t - 1]
                    action_duriation -= 1
                    action_selection[index_sample, t - 1] = action_random_selection
                else:
                    action_selection[index_sample, t - 1] = 4
                    position[index_sample, :, t] = position[index_sample, :, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True

    for index_sample in range(model.hparams.train.batch_size):
        action_one_hot_value_numpy[
            index_sample, action_selection[index_sample], np.array(range(model.hparams.gtm_sm.total_dim - 1))] = 1

    action_one_hot_value = torch.from_numpy(action_one_hot_value_numpy).to(device=model.device)

    return action_one_hot_value, position, action_selection


def sample_position(model):
    for params in model.hparams.enc_st_matrix.parameters():
        enc_st_matrix_params = params

    right_vector = enc_st_matrix_params[:, 0].detach().cpu().numpy()
    up_vector  = enc_st_matrix_params[:, 2].detach().cpu().numpy()

    project_matrix = np.hstack((right_vector.reshape((2, 1)), up_vector.reshape((2, 1))))

    x, y = np.meshgrid(np.arange(-8.,9., dtype=np.float32), np.arange(-8.,9., dtype=np.float32))
    pos_index = np.vstack((x.reshape((1, -1)), y.reshape((1, -1))))

    X_train  = np.dot(project_matrix, pos_index).T
    Y_train = np.zeros((X_train.shape[0], 1), np.float32)
    true_index = np.logical_and(pos_index[0] <= 4.5, pos_index[0] >= -4.5)
    true_index = np.logical_and(true_index, pos_index[1] <= 4.5)
    true_index = np.logical_and(true_index, pos_index[1] >= -4.5)
    Y_train[true_index,:] = 1

    X_train = torch.from_numpy(X_train).to(device=model.device)
    Y_train = torch.from_numpy(Y_train).to(device=model.device)

    return X_train, Y_train
