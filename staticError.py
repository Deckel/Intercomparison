import numpy as np
from netCDF4 import Dataset
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import glob
import fnmatch
import math
import pandas as pd


# Get CORE FAAM data
def getDataFaam():
	fnames = glob.glob("/media/faamarchive/Documents/Instruments/Uncertainties/ps_rvsm - SEPnet2018/EMERGE_data/core_faam_20170713_v004_rJOE_c021.nc")
	for i in fnames:
		fh = Dataset(i,'r')
	variables = ["Time", "PS_RVSM", "ALT_GIN", "IAS_RVSM", "PALT_RVS", "TAT_DI_R", "ROLL_GIN", "PSP_TURB", "TAS_RVSM","Q_RVSM"]
	dfFaam = pd.DataFrame(columns = variables)
	for i in variables:
		var = fh.variables[i][:]
		var = var.ravel()
		if i != "Time":
			var = var[0::32]
		dfFaam[i] = var
	dfFaam = dfFaam.rename(columns = {"Time":"TIME"})
	return(dfFaam)

# Get BAHAMAS data
def getDataDLR():
	variables = ["TIME", "PS", "QC", "IRS_ALT", "TAS", "IRS_PHI", "TAT"]
	dfDLR = pd.DataFrame(columns = variables)
	with open("/media/faamarchive/Documents/Instruments/Uncertainties/ps_rvsm - SEPnet2018/EMERGE_data/EMERGE-04_2017-07-13_BAHAMAS_V1.ames") as f:
		textRaw = [line.split() for line in f]	
		data = textRaw[152:]
		for i in variables:
			var = []
			for j in data[1:]:
				var.append(j[data[0].index(i)])
			dfDLR[i] = np.array(var, dtype = float)
	return(dfDLR)

# Cut irrelevant data
def cutData(df):
	df = pd.concat([df[int(df[df["TIME"]==40800.0].index.values):int(df[df["TIME"]==41800.0].index.values)],
					df[int(df[df["TIME"]==42128.0].index.values):int(df[df["TIME"]==43596.0].index.values)],
					df[int(df[df["TIME"]==44030.0].index.values):int(df[df["TIME"]==46252.0].index.values)]], ignore_index=True)
	return(df)

# Corrects for residual pressure due to altitude diffrence 
def altCorrection(df):
	#Residual altitde difference in meters
	df["residualAlt"] = df["IRS_ALT"] - df["ALT_GIN"]
	#Density of air by avaraging FAAM and HALO aircraft
	rho = ( df["PS_RVSM"]/(287.05*df["TAT_DI_R"]) + df["PS"]/(287.05*df["TAT"]) ) / 2
	#Residual pressure
	df["residualPressure"] = rho*9.80665*df["residualAlt"]
	#Apply Correction
	df["PS_RVSM_ALTCORR"] = df["PS_RVSM"] - df["residualPressure"]
# Corrects for position Error using BAE -903 Law
def positionErrorCorrection(df):
	# Calculate residual model pressure and convert IAS and PALT to kts and ft respectivly
	df["residualModelPressure"] = (0.0404*df["IAS_RVSM"]*1.94) + (0.0000297*df["PALT_RVS"]*3.2084) - 8.13
	df["PS_RVSM_CORR"] = df["PS_RVSM_ALTCORR"] - df["residualModelPressure"]
# Removes data where roll angle was to high for accurate pressure reading += 5 deg for both FAAM & HALO
def removeRoll(df):
	return(df[(df["ROLL_GIN"] <= 5) & (df["ROLL_GIN"] >= -5) & (df["IRS_PHI"] <= 5) & (df["IRS_PHI"] >= -5)])
# Calculate mach number for both aircraft
def mach(df):
    df["machFAAM"] = np.sqrt(5.0 * ((1.0 + df["Q_RVSM"] / df["PS_RVSM"])**(2.0 / 7.0) - 1.0))
    df["machHALO"] = np.sqrt(5.0 * ((1.0 + df["QC"] / df["PS"])**(2.0 / 7.0) - 1.0))
# Apply all error corrections
def errorCorrection(df):
	df["error"] = df["PS_RVSM_ALTCORR"] - df["PS"]
	fit = np.poly1d(np.polyfit(df["machFAAM"], df["error"],1))
	df["PS_RVSM_MODCORR"] = df["PS_RVSM_ALTCORR"] - fit(df["machFAAM"])
	df["errorModCorrected"] = df["PS_RVSM_MODCORR"] - df["PS"]
	df["error903Corrected"] = df["PS_RVSM_CORR"] - df["PS"]
	return(fit)
 
 # Main Processing
dfFaam = getDataFaam()
dfDLR = getDataDLR()
df = pd.merge(dfFaam, dfDLR, on=["TIME"])
df = cutData(df)
altCorrection(df)
positionErrorCorrection(df)
df  = removeRoll(df)
mach(df)

# Calculate fit from M = 0.32 to M = 0.42 for the correction
fit = errorCorrection(df)
print("Equation for the correction derrived by the comparison and the uncertainty is this correction was applied")
print("y = {0:.2f}x + {0:.2f}".format(*fit.c))
print(np.std(df["errorModCorrected"]))


""" Exmaple plot, with the data frame you can plot what you need:
KEY: error             = The error after the altitude correction has been applied
	 error903Corrected = The error after altitude correction and 903 law resudials have been applied 
	 errorModCorrected = The error after the derrived fit from the inter-comparison (Needs to be applied to other comparison data to see if it works)

	 PS_RVSM           = S2 Pressure
	 PS_RVSM_ALTCORR   = Pressure afetr the pressure altitude difference between the planes has been subtracted
	 PS_RVSM_CORR      = Pressure after the 903 law has been applied to PS_RVSM_ALTCORR
	 PS_RVSM_MODCORR   = Pressure after the derrived correction (from this intercomparison) has been applied to PS_RVSM_ALTCORR

	 Variable key can be found in the respective folders
"""
fig, ax1 = plt.subplots()
ax1.scatter(df["TIME"], df["error"], s=0.5, c="red", alpha = 0.3, label="uncorrected")
ax1.scatter(df["TIME"], df["error903Corrected"], s=0.5, c="blue", label = "corrected")
ax1.set_xlabel("Mach Number", fontsize = 13)
ax1.set_ylabel("Residual Error /Hpa FAAM-HALO", fontsize = 13)
ax1.set_title("Position error as a function of Mach in the FAAM aircraft")
ax1.axhline(0, c="black")
ax1.legend(loc="upper left")
ax1.text(0.4, -2, "y = {0:.2f}x + {0:.2f}".format(*fit.c))
plt.show()




