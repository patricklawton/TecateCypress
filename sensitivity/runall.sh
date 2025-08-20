mpiexec -np 7 python phase_uncertain.py
python postprocess.py
python makefig.py
python interp_optima_uncertain.py
python make_interp_plots.py
mpiexec -np 2 python phase_optdecisions.py
python uncertainty_pairplots.py
