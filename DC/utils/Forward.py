import warnings 
import numpy as np
from discretize import TensorMesh
from discretize.utils import mkvc

from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import generate_dcip_sources_line

from SimPEG import maps

try:
    from pymatsolver import Pardiso as Solver
    
except ImportError:
    from SimPEG import SolverLU as Solver

def Forward(model=None, surveyinfo = None):

    if model is None:
        warnings.warn('Warning Message: Ingrese el modelo, mi perro')

    if surveyinfo is None:
        warnings.warn('Warning Message: Ingrese el la información del survey, mi perro. ¿Qué pasa, mi rey?')

    model = model[::-1]
    nrow, ncol = model.shape
    nElec = surveyinfo["nElec"]
    sep = surveyinfo["sep"]

    pi = surveyinfo["pi"] # punto inicial 
    pf = sep*nElec # punto final

    survey_type = surveyinfo["typ_survey"] # Tipo arreglo
    dimension_type = surveyinfo["dim"]
    data_type = surveyinfo["data_type"] # Tipo de datos: voltajes, resistividades, conductividades

    end_locations = np.r_[pi, pf] # Posiciones de los electrodos (xi, xf)
    station_separation = sep # Separación 
    num_rx_per_src = surveyinfo["nlines"] # Número máximo de líneas (n)

    # Definición de topografía en función del arreglo
    x = np.linspace(pi, pf, int(pf+1)) # el numero de elementos debe ser > a los elementos en el eje X de la malla
    z = np.linspace(-pf/3, 0, int(pf/3)+1)
    x_topo, z_topo = np.meshgrid(x, z)
    topo_xz = np.c_[mkvc(x_topo), mkvc(z_topo)]

    topo_2d = np.unique(topo_xz[:, [0, 1]], axis=0)

    # Generate source list for DC survey line
    source_list = generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations,
        topo_2d,
        num_rx_per_src,
        station_separation,
    )

    # Define survey
    survey = dc.survey.Survey(source_list, survey_type=survey_type) # Tipo de arreglo; # electrodos; Total de datos

    if surveyinfo["data_type"] == "apparent_resistivity":
        survey.set_geometric_factor()

    l = np.diff(end_locations)[0] # Longitud del arreglo

    dh = l/ncol # Tamaño de celda x 
    dz = surveyinfo["depth"]/nrow # Tamaño de celda y
    hx = [(dh, ncol)] # [(Tamaño de celda, número de celdas)]
    hz = [(dz, nrow)]
    mesh = TensorMesh([hx, hz], "0N") # 0: centrado en el origen; N: Inicio de eje negativo: C: 0 como centro 

    model = np.log(model.flatten())

    mapping = maps.ExpMap(mesh)

    simulation_dc = dc.Simulation2DNodal(
        mesh, rhoMap=mapping, solver=Solver, survey=survey, storeJ=True, nky=12
    )

    dc_data = simulation_dc.make_synthetic_data(model, relative_error=0, add_noise=False)

    return mapping, survey, mesh, simulation_dc, dc_data