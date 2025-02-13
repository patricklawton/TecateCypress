import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from sbi import utils as utils
from sbi.inference import simulate_for_sbi
from sbi.diagnostics.lc2st import LC2ST
from sbi.analysis.plot import pp_plot_lc2st
from sbi.analysis.plot import marginal_plot_with_probs_intensity
from sbi.utils.analysis_utils import get_probs_per_marginal
import os

overwrite = True
processes = ['fecundity']
for pr in processes:
    if pr == 'mortality':
        from mortality.simulator import simulator, fixed
        #labels = ['alph_m', 'beta_m', 'sigm_m', 'gamm_nu', 'kappa', 'K_adult']
        NUM_CAL = 3_000
    elif pr == 'fecundity':
        from fecundity.simulator import simulator, fixed
        #fixed = {}
        #labels = ['rho_max', 'eta_rho', 'a_mature', 'sigm_max', 'eta_sigm']
        NUM_CAL = 1_000
    with open(pr+"/restricted_prior.pkl", "rb") as handle:
        prior = pickle.load(handle)
    simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

    with open(pr+"/param_labels.pkl", 'rb') as handle:
        labels = pickle.load(handle)

    with open(pr+"/posterior.pkl", "rb") as handle:
        posterior = pickle.load(handle)

    NUM_EVAL = 15_000
    num_ref_obs = 9

    if overwrite:
        # get reference observations
        torch.manual_seed(0) # seed for reproducibility
        thetas_star, xs_star = simulate_for_sbi(simulator, proposal=prior, num_simulations=100, num_workers=1)
        # filter out invalid parameter samples
        thetas_star = thetas_star[~torch.any(xs_star.isnan(),dim=1)]
        xs_star = xs_star[~torch.any(xs_star.isnan(),dim=1)]
        # take the first few valid
        thetas_star = thetas_star[:num_ref_obs]
        xs_star = xs_star[:num_ref_obs]

        # Sample from the estimated posterior
        post_samples_star = [posterior.sample(sample_shape=(NUM_EVAL,), x=xs_star[i]) for i in range(num_ref_obs)]
        post_samples_star = torch.stack(post_samples_star, dim=0)

        with open(pr + '/lc2st_data/thetas_star.pkl', 'wb') as handle:
            pickle.dump(thetas_star, handle)
        with open(pr + '/lc2st_data/xs_star.pkl', 'wb') as handle:
            pickle.dump(xs_star, handle)
        with open(pr + '/lc2st_data/post_samples_star.pkl', 'wb') as handle:
            pickle.dump(post_samples_star, handle)

        torch.manual_seed(42) # seed for reproducibility

        # sample calibration data
        theta_cal, x_cal = simulate_for_sbi(simulator, proposal=prior, num_simulations=NUM_CAL, num_workers=8)
        # filter out invalid parameter samples
        theta_cal = theta_cal[~torch.any(x_cal.isnan(),dim=1)]
        x_cal = x_cal[~torch.any(x_cal.isnan(),dim=1)]
        post_samples_cal = [posterior.sample(sample_shape=(1,), x=xc)[0] for xc in x_cal]
        post_samples_cal = torch.stack(post_samples_cal, dim=0)

        with open(pr + '/lc2st_data/theta_cal.pkl', 'wb') as handle:
            pickle.dump(theta_cal, handle)
        with open(pr + '/lc2st_data/x_cal.pkl', 'wb') as handle:
            pickle.dump(x_cal, handle)
        with open(pr + '/lc2st_data/post_samples_cal.pkl', 'wb') as handle:
            pickle.dump(post_samples_cal, handle)

        # set up the LC2ST: train the classifiers
        lc2st = LC2ST(
            thetas=theta_cal,
            xs=x_cal,
            posterior_samples=post_samples_cal,
            classifier="mlp",
            num_ensemble=1, # number of classifiers for the ensemble
        )
        _ = lc2st.train_under_null_hypothesis() # over 100 trials under (H0)
        _ = lc2st.train_on_observed_data() # on observed data
        # save trained classifier to pkl
        with open(pr+'/lc2st_data/lc2st_classifier.pkl', 'wb') as handle:
            pickle.dump(lc2st, handle)

    else:
        # Read in existing data
        with open(pr + '/lc2st_data/thetas_star.pkl', 'rb') as handle:
            thetas_star = pickle.load(handle)
        with open(pr + '/lc2st_data/xs_star.pkl', 'rb') as handle:
            xs_star = pickle.load(handle)
        with open(pr + '/lc2st_data/post_samples_star.pkl', 'rb') as handle:
            post_samples_star = pickle.load(handle)
        with open(pr + '/lc2st_data/theta_cal.pkl', 'rb') as handle:
            theta_cal = pickle.load(handle)
        with open(pr + '/lc2st_data/x_cal.pkl', 'rb') as handle:
            x_cal = pickle.load(handle)
        with open(pr + '/lc2st_data/post_samples_cal.pkl', 'rb') as handle:
            post_samples_cal = pickle.load(handle)
        with open(pr+'/lc2st_data/lc2st_classifier.pkl', 'rb') as handle:
            lc2st = pickle.load(handle)


    # Define significance level for diagnostics
    conf_alpha = 0.05

    fig, axes = plt.subplots(1,len(thetas_star), figsize=(12*(num_ref_obs/3),3))
    for i in range(len(thetas_star)):
        probs, scores = lc2st.get_scores(
            theta_o=post_samples_star[i],
            x_o=xs_star[i],
            return_probs=True,
            trained_clfs=lc2st.trained_clfs
        )
        T_data = lc2st.get_statistic_on_observed_data(
            theta_o=post_samples_star[i],
            x_o=xs_star[i]
        )
        T_null = lc2st.get_statistics_under_null_hypothesis(
            theta_o=post_samples_star[i],
            x_o=xs_star[i]
        )
        p_value = lc2st.p_value(post_samples_star[i], xs_star[i])
        reject = lc2st.reject_test(post_samples_star[i], xs_star[i], alpha=conf_alpha)

        # plot 95% confidence interval
        quantiles = np.quantile(T_null, [0, 1-conf_alpha])
        axes[i].hist(T_null, bins=50, density=True, alpha=0.5, label="Null")
        axes[i].axvline(T_data, color="red", label="Observed")
        axes[i].axvline(quantiles[0], color="black", linestyle="--", label="95% CI")
        axes[i].axvline(quantiles[1], color="black", linestyle="--")
        axes[i].set_xlabel("Test statistic")
        axes[i].set_ylabel("Density")
        #axes[i].set_xlim(-0.01,0.25)
        axes[i].set_title(
            f"observation {i+1} \n p-value = {p_value:.3f}, reject = {reject}"
        )
    axes[-1].legend(bbox_to_anchor=(1.1, .5), loc='center left')
    #plt.show()
    fig.savefig(f'sbi_figs/{pr}_lc2st_1.png', bbox_inches='tight')

    # P-P plots

    fig, axes = plt.subplots(1,len(thetas_star), figsize=(12*(num_ref_obs/3),3))
    for i in range(len(thetas_star)):
        probs_data, _ = lc2st.get_scores(
            theta_o=post_samples_star[i],
            x_o=xs_star[i],
            return_probs=True,
            trained_clfs=lc2st.trained_clfs
        )
        probs_null, _ = lc2st.get_statistics_under_null_hypothesis(
            theta_o=post_samples_star[i],
            x_o=xs_star[i],
            return_probs=True
        )

        pp_plot_lc2st(
            probs=[probs_data],
            probs_null=probs_null,
            conf_alpha=conf_alpha,
            labels=["Classifier probabilities \n on observed data"],
            colors=["red"],
            ax=axes[i],
        )
        axes[i].set_title(f"PP-plot for observation {i+1}")
    axes[-1].legend(bbox_to_anchor=(1.1, .5), loc='center left')
    fig.savefig(f'sbi_figs/{pr}_lc2st_2.png', bbox_inches='tight')


    label = "Probabilities (class 0)"
    # label = r"$\hat{p}(\Theta\sim q_{\phi}(\theta \mid x_0) \mid x_0)$"

    fig, axes = plt.subplots(len(thetas_star), 7, figsize=(9.5*2,6*(num_ref_obs/3)), constrained_layout=True)
    for i in range(len(thetas_star)):
        probs_data, _ = lc2st.get_scores(
            theta_o=post_samples_star[i][:1000],
            x_o=xs_star[i],
            return_probs=True,
            trained_clfs=lc2st.trained_clfs
        )
        dict_probs_marginals = get_probs_per_marginal(
            probs_data[0],
            post_samples_star[i][:1000].numpy()
        )
        # 2d histogram
        marginal_plot_with_probs_intensity(
            dict_probs_marginals['0_1'],
            marginal_dim=2,
            ax=axes[i][0],
            n_bins=50,
            label=label
        )

        for j in range(len(labels)):
            # marginal 1
            marginal_plot_with_probs_intensity(
                dict_probs_marginals[f'{j}'],
                marginal_dim=1,
                ax=axes[i][j+1],
                n_bins=50,
                label=label,
            )

        ## marginal 1
        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['0'],
        #    marginal_dim=1,
        #    ax=axes[i][1],
        #    n_bins=50,
        #    label=label,
        #)

        ## marginal 1
        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['0'],
        #    marginal_dim=1,
        #    ax=axes[i][1],
        #    n_bins=50,
        #    label=label,
        #)

        ## marginal 2
        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['1'],
        #    marginal_dim=1,
        #    ax=axes[i][2],
        #    n_bins=50,
        #    label=label,
        #)

        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['2'],
        #    marginal_dim=1,
        #    ax=axes[i][3],
        #    n_bins=50,
        #    label=label,
        #)

        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['3'],
        #    marginal_dim=1,
        #    ax=axes[i][4],
        #    n_bins=50,
        #    label=label,
        #)

        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['4'],
        #    marginal_dim=1,
        #    ax=axes[i][5],
        #    n_bins=50,
        #    label=label,
        #)

        #marginal_plot_with_probs_intensity(
        #    dict_probs_marginals['5'],
        #    marginal_dim=1,
        #    ax=axes[i][6],
        #    n_bins=50,
        #    label=label,
        #)

    axes[0][1].set_title("marginal 1")
    axes[0][2].set_title("marginal 2")

    for j in range(num_ref_obs):
        axes[j][0].set_ylabel(f"observation {j + 1}")
    axes[0][2].legend()
    fig.savefig(f'sbi_figs/{pr}_lc2st_3.png', bbox_inches='tight')
