from project import Phase
import numpy as np
from threadpoolctl import threadpool_limits

# First, specify which operations to run
run_baseline = False
run_uncertain = False
run_interp = False
run_optdec = True

# The following constants will be shared across all results to follow
constants = {}
constants['c'] = 1.42
constants['Aeff'] = 7.29
constants['t_final'] = 300
constants['sim_method'] = 'discrete'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
'''Should just set this to min tau_vec I think, which it basically already is'''
constants['min_tau'] = 2
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['ncell_min'] = 2_500
constants['root'] = 0 #For mpi
constants['final_max_tau'] =  np.nan
constants['meta_metric'] = 'gte_thresh'
metric_thresh = 0.975 # Threshold of pop metric value used for calculating meta metric
constants['metric'] = 'lambda_s'
constants['overwrite_metrics'] = False

# Now specify sample densities and parameer bounds for running baseline phase
constants['tauc_min_samples'] = np.arange(1,15,1)
constants['ncell_samples'] = 100
constants['slice_samples'] = 200
constants['num_eps_combs'] = 1
# Theoretical (or ad-hoc) maxima/minima for parameters
'''Demo sample index 0 is reserved for MAP lambda(tau)'''
minima = {
    'mu_tau': 0.,
    'sigm_tau': 0.,
    'mu_tauc': 0,
    'sigm_tauc': 0.,
    'demographic_index': 0
}
maxima = {
    'mu_tau': 0.,
    'sigm_tau': 0,
    'mu_tauc': 0.,
    'sigm_tauc': 0,
    'demographic_index': 0
}

# Process baseline samples
pproc = Phase(**constants)
pproc.initialize()
pproc.init_decision_parameters(overwrite=True, suffix="_baseline")
if run_baseline:
    pproc.process_samples(minima, maxima, "_baseline", metric_thresh)

# Now specify sample densities and parameter bounds for running uncertain phase
constants.update({'tauc_min_samples': np.arange(2, 18, 4)})
constants.update({'ncell_samples': 30})
constants.update({'slice_samples': 55})
constants.update({'num_eps_combs': 20_000}) #Uncertainty samples per decision
constants.update({'overwrite_metrics': False}) #Dont rerun metric initialization stuff
minima.update({'mu_tau': -0.75})
minima.update({'sigm_tau': 0.})
minima.update({'mu_tauc': -0.75})
minima.update({'sigm_tauc': 0.})
minima.update({'demographic_index': 1})
maxima.update({'mu_tau': 0.15})
maxima.update({'sigm_tau': 0.2})
maxima.update({'mu_tauc': 0.15})
maxima.update({'sigm_tauc': 0.2})
maxima.update({'demographic_index': pproc.num_demographic_samples - 1})

# Process samples under uncertainty
pproc = Phase(**constants)
pproc.initialize()
pproc.init_decision_parameters(overwrite=True, suffix="_uncertain")
if run_uncertain:
    pproc.process_samples(minima, maxima, "_uncertain", metric_thresh)

# Select a value of total resources to focus on going forward
taucmin = 6 #This is just a value of C/ncell_tot

# Do some postprocessing on the phase results on root processor
pproc.comm.Barrier()
if pproc.rank == pproc.root:
    # Allow use of all cores for this step
    with threadpool_limits(limits=None):
        pproc.postprocess_phase_uncertain()
        pproc.compute_nochange(metric_thresh) #Compute result under no change for reference later

        # Interpolate optima under baseline conditions
        nn = 100 #Nearest neighbors used by RBF interpolator
        if run_interp:
            print("Optimizing basline")
            pproc.interp_optima_baseline(taucmin, nn)

        # Interpolate optimally robust decisions
        nn = 150
        smoothing = 0.01 #Used by RBF to smooth noisy robustness samples
        num_restarts = 8 #Number of starting points to consider in optimization
        if run_interp:
            print("Optimizing robustness")
            pproc.interp_optima_uncertain(taucmin, nn, smoothing, num_restarts)
pproc.comm.Barrier()

# Read in things created above we want to reference below
S_opt_baseline = np.load(pproc.data_dir + '/S_opt_baseline.npy')
decision_opt_baseline = np.load(pproc.data_dir + '/decision_opt_baseline.npy')
Sstar_vec = np.load(pproc.data_dir + '/Sstar_vec.npy')
decision_opt_uncertain = np.load(pproc.data_dir + '/decision_opt_uncertain.npy')

# Get optimal decisions at a few values of q (% decrease in S_opt_baseline)
q_vec = np.array([0.0, 0.125, 0.25, 0.37450, 0.5, 0.625]) 
n_opt = np.empty(len(q_vec) + 1)
l_opt = np.empty(len(q_vec) + 1)
Sstar_i_optdecisions = np.empty(len(q_vec))

# First, get optima under baseline conditions
n_opt[0] = decision_opt_baseline[0]
l_opt[0] = decision_opt_baseline[1]

# Now get the optimal decisions for (1-q) * optimal S baseline
for q_i, q in enumerate(q_vec):
    Sstar_i = np.argmin(np.abs(Sstar_vec - ((1 - q) * S_opt_baseline)) )
    Sstar_i_optdecisions[q_i] = Sstar_i
    n_opt[q_i + 1] = decision_opt_uncertain[Sstar_i, 0]
    l_opt[q_i + 1] = decision_opt_uncertain[Sstar_i, 1]

    # Replace this q value with the closest one we have available
    '''This may not be necessary'''
    Sstar = Sstar_vec[Sstar_i]
    q_vec[q_i] = 1 - (Sstar / S_opt_baseline)

# Save q_vec and associated Sstar indices
if pproc.rank == pproc.root:
    np.save(pproc.data_dir + '/q_vec.npy', q_vec)
    np.save(pproc.data_dir + '/Sstar_i_optdecisions.npy', Sstar_i_optdecisions.astype(int))

# Now update decision parameters and number of uncertainty samples
constants.update({'ncell_samples': n_opt})
constants.update({'slice_samples': l_opt})
constants.update({'tauc_min_samples': np.repeat(taucmin, len(q_vec) + 1)})
constants.update({'num_eps_combs': 5_000_000})

# Now process many uncertainty samples at the optimal decisions
pproc = Phase(**constants)
pproc.initialize()
pproc.init_decision_parameters(overwrite=True, suffix='_optdecisions')
if run_optdec:
    pproc.process_samples(minima, maxima, "_optdecisions", metric_thresh)
