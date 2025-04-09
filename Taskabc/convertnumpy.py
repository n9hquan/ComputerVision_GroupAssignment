import json
import numpy as np

with open("output\camera_params_PA.json", "r") as f:
    params = json.load(f)

cameraMatrix1 = np.array(params["camera_matrix_left"])
distCoeffs1 = np.array(params["dist_coeff_left"])
cameraMatrix2 = np.array(params["camera_matrix_right"])
distCoeffs2 = np.array(params["dist_coeff_right"])
R = np.array(params["R"])
T = np.array(params["T"])
