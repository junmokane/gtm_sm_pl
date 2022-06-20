import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_experiment_information(model, x, st_observation_list, st_prediction_list, xt_prediction_list, position, fname):
    
    sample_id = np.random.randint(0, model.hparams.train.batch_size, size=(1))
    sample_imgs = x[sample_id]

    st_observation_sample = np.zeros((model.hparams.gtm_sm.observe_dim, model.hparams.gtm_sm.s_dim))
    for t in range(model.hparams.gtm_sm.observe_dim):
        st_observation_sample[t] = st_observation_list[t][sample_id].cpu().detach().numpy()

    st_prediction_sample = np.zeros((model.hparams.gtm_sm.total_dim - model.hparams.gtm_sm.observe_dim, model.hparams.gtm_sm.s_dim))
    for t in range(model.hparams.gtm_sm.total_dim - model.hparams.gtm_sm.observe_dim):
        st_prediction_sample[t] = st_prediction_list[t][sample_id].cpu().detach().numpy()

    st_2_max = np.maximum(np.max(st_observation_sample[:, 0]), np.max(st_prediction_sample[:, 0]))
    st_2_min = np.minimum(np.min(st_observation_sample[:, 0]), np.min(st_prediction_sample[:, 0]))
    st_1_max = np.maximum(np.max(st_observation_sample[:, 1]), np.max(st_prediction_sample[:, 1]))
    st_1_min = np.minimum(np.min(st_observation_sample[:, 1]), np.min(st_prediction_sample[:, 1]))
    axis_st_1_max = st_1_max + (st_1_max - st_1_min) / 10.0
    axis_st_1_min = st_1_min - (st_1_max - st_1_min) / 10.0
    axis_st_2_max = st_2_max + (st_2_max - st_2_min) / 10.0
    axis_st_2_min = st_2_min - (st_2_max - st_2_min) / 10.0

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    fig.clf()
    gs = fig.add_gridspec(2, 2)

    # plot position trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Observation (256 steps)')
    ax1.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, :model.hparams.gtm_sm.observe_dim].T, position[sample_id, 0, :model.hparams.gtm_sm.observe_dim].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, model.hparams.gtm_sm.observe_dim-1], position[sample_id, 0, model.hparams.gtm_sm.observe_dim-1], 'bs')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Prediction (256 steps)')
    ax2.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, model.hparams.gtm_sm.observe_dim:].T, position[sample_id, 0, model.hparams.gtm_sm.observe_dim:].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, -1], position[sample_id, 0, -1], 'bs')

    # plot inferred states
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlabel('$s_1$')
    ax3.set_ylabel('$s_2$')
    ax3.set_title('Inferred states')
    ax3.set_aspect('equal')
    plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
    plt.gca().invert_yaxis()
    plt.plot(st_observation_sample[0:, 1].T, st_observation_sample[0:, 0].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(st_observation_sample[-1, 1], st_observation_sample[-1, 0], 'bs')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlabel('$s_1$')
    ax4.set_ylabel('$s_2$')
    ax4.set_title('Inferred states')
    ax4.set_aspect('equal')
    plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
    plt.gca().invert_yaxis()
    plt.plot(st_prediction_sample[0:, 1].T, st_prediction_sample[0:, 0].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(st_prediction_sample[-1, 1], st_prediction_sample[-1, 0], 'bs')
    
    plt.savefig(f'{fname}/exp_traj.png')
    