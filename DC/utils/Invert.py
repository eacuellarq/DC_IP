import numpy as np
import warnings 

from SimPEG.electromagnetics.static.utils.static_utils import apparent_resistivity_from_voltage
    
from SimPEG import (
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)

from utils.results_inversion import SaveInversionProgress

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def Invert(floor=1e-3, percent_std=0.05, alpha_s=1e-3, 
           alpha_x=1, alpha_y=1, iter=20, b0_ratio=1e2, use_target=False,
           CF=2., CR=1, dc_data=None, simulation_dc=None, survey=None, mesh=None):

    floor = floor
    percent_std = percent_std

    dc_data.standard_deviation = floor + percent_std * np.abs(dc_data.dobs)

    mref = np.log(np.median(apparent_resistivity_from_voltage(survey,dc_data.dobs)))*np.ones(mesh.nC)
    dmisfit = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation_dc)
    reg = regularization.WeightedLeastSquares(
        mesh, alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, reference_model=mref*np.ones(mesh.nC)
    )

    opt = optimization.InexactGaussNewton(maxIter=iter, maxIterCG=iter)
    opt.remember("xc")
    base_inv = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

    beta_est = directives.BetaEstimate_ByEig(beta0_ratio=b0_ratio)
    target = directives.TargetMisfit(chifact=1)
    save = SaveInversionProgress()
    directives_list = [beta_est, save]

    if use_target is True:
        directives_list.append(target)

    beta_schedule = directives.BetaSchedule(coolingFactor=CF, coolingRate=CR)
    directives_list.append(beta_schedule)

    inv = inversion.BaseInversion(base_inv, directiveList=directives_list)
    model_recovered = inv.run(mref)
    inv_result = save.inversion_results

    return inv_result 
