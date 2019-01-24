import uproot
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy import stats
from xgbo import XgboRegressor
import os

def load_data(file_name, entrystop=None, isEE=False):

    root_file = uproot.open(file_name)

    # The branches we need for the regression
    branches_EB = [ 'clusterRawEnergy', 'full5x5_e3x3', 'full5x5_eMax',
            'full5x5_e2nd', 'full5x5_eTop', 'full5x5_eBottom', 'full5x5_eLeft',
            'full5x5_eRight', 'full5x5_e2x5Max', 'full5x5_e2x5Top',
            'full5x5_e2x5Bottom', 'full5x5_e2x5Left', 'full5x5_e2x5Right',
            'full5x5_e5x5', 'rawEnergy', 'etaWidth', 'phiWidth', 'rhoValue',
            'full5x5_sigmaIetaIeta', 'full5x5_sigmaIetaIphi',
            'full5x5_sigmaIphiIphi', 'iEtaSeed', 'iPhiSeed', 'iEtaMod5',
            'iPhiMod2', 'iEtaMod20', 'iPhiMod20', 'genEnergy']

    branches_EE = [ 'clusterRawEnergy', 'full5x5_e3x3', 'full5x5_eMax',
            'full5x5_e2nd', 'full5x5_eTop', 'full5x5_eBottom', 'full5x5_eLeft',
            'full5x5_eRight', 'full5x5_e2x5Max', 'full5x5_e2x5Top',
            'full5x5_e2x5Bottom', 'full5x5_e2x5Left', 'full5x5_e2x5Right',
            'full5x5_e5x5', 'rawEnergy', 'etaWidth', 'phiWidth', 'rhoValue',
            'full5x5_sigmaIetaIeta', 'full5x5_sigmaIetaIphi',
            'full5x5_sigmaIphiIphi',
            'genEnergy', 'iXSeed', 'iYSeed', 'preshowerEnergy']

    if isEE:
        branches = branches_EE + ["pt", "eta"]
    else:
        branches = branches_EB + ["pt", "eta"]

    if "Electron" in file_name:
        df = root_file['een_analyzer/ElectronTree'].pandas.df(branches, entrystop=entrystop).dropna()
    if "Photon" in file_name:
        df = root_file['een_analyzer/PhotonTree'].pandas.df(branches, entrystop=entrystop).dropna()
    print("Entries in ntuple:")
    print(len(df))

    # Define some ratio variables
    df.eval("clusertRawEnergyOverE5x5 = clusterRawEnergy/full5x5_e5x5", inplace=True)
    df.eval("w3x3OverE5x5             = full5x5_e3x3/full5x5_e5x5", inplace=True)
    df.eval("eMaxOverE5x5             = full5x5_eMax/full5x5_e5x5", inplace=True)
    df.eval("e2ndOverE5x5             = full5x5_e2nd/full5x5_e5x5", inplace=True)
    df.eval("eTopOverE5x5             = full5x5_eTop/full5x5_e5x5", inplace=True)
    df.eval("eBottomOverE5x5          = full5x5_eBottom/full5x5_e5x5", inplace=True)
    df.eval("eLeftOverE5x5            = full5x5_eLeft/full5x5_e5x5", inplace=True)
    df.eval("eRightOverE5x5           = full5x5_eRight/full5x5_e5x5", inplace=True)
    df.eval("e2x5MaxOverE5x5          = full5x5_e2x5Max/full5x5_e5x5", inplace=True)
    df.eval("e2x5TopOverE5x5          = full5x5_e2x5Top/full5x5_e5x5", inplace=True)
    df.eval("e2x5BottomOverE5x5       = full5x5_e2x5Bottom/full5x5_e5x5", inplace=True)
    df.eval("e2x5LeftOverE5x5         = full5x5_e2x5Left/full5x5_e5x5", inplace=True)
    df.eval("e2x5RightOverE5x5        = full5x5_e2x5Right/full5x5_e5x5", inplace=True)

    if isEE:
        df.eval("preshowerEnergyOverrawEnergy = preshowerEnergy/rawEnergy", inplace=True)

    # The target
    if isEE:
        df.eval("target = genEnergy / ( rawEnergy + preshowerEnergy )", inplace=True)
    else:
        df.eval("target = genEnergy / rawEnergy", inplace=True)

    return df

# The features
features_EB = [ 'rawEnergy', 'etaWidth', 'phiWidth', 'rhoValue',
        'full5x5_sigmaIetaIeta', 'full5x5_sigmaIetaIphi',
        'full5x5_sigmaIphiIphi', 'clusertRawEnergyOverE5x5', 'w3x3OverE5x5',
        'eMaxOverE5x5', 'e2ndOverE5x5', 'eTopOverE5x5', 'eBottomOverE5x5',
        'eLeftOverE5x5', 'eRightOverE5x5', 'e2x5MaxOverE5x5',
        'e2x5TopOverE5x5', 'e2x5BottomOverE5x5', 'e2x5LeftOverE5x5',
        'e2x5RightOverE5x5', 'iEtaSeed', 'iPhiSeed', 'iEtaMod5', 'iPhiMod2',
        'iEtaMod20', 'iPhiMod20']

# EE
features_EE = [ 'rawEnergy', 'etaWidth', 'phiWidth', 'rhoValue',
        'full5x5_sigmaIetaIeta', 'full5x5_sigmaIetaIphi',
        'full5x5_sigmaIphiIphi', 'clusertRawEnergyOverE5x5', 'w3x3OverE5x5',
        'eMaxOverE5x5', 'e2ndOverE5x5', 'eTopOverE5x5', 'eBottomOverE5x5',
        'eLeftOverE5x5', 'eRightOverE5x5', 'e2x5MaxOverE5x5',
        'e2x5TopOverE5x5', 'e2x5BottomOverE5x5', 'e2x5LeftOverE5x5',
        'e2x5RightOverE5x5', 'iXSeed', 'iYSeed', 'preshowerEnergyOverrawEnergy']

#file_name = "/scratch/micheli/Electron/perfectIC-highpt-EB-training.root"
file_name = "/scratch/micheli/Electron/perfectIC-lowpt-EB-training.root"

isEE = '-EE-' in file_name

if isEE:
    features = features_EE
else:
    features = features_EB

tmp = file_name.split("/")
out_dir = tmp[-2] + "_" + tmp[-1].replace("-training.root", "_test20190124")

def rmseff(x, c=0.683):
    try:
        x = np.sort(x, kind='mergesort')
        m = int(c*len(x)) + 1
        return np.min(x[m:] - x[:-m])/2.0
    except:
        return np.nan

def print_pcolor_labels(ax, X, Y, C):
    for x1, x2, y1, y2, c in zip(
                       X[:-1,:-1].flatten(),
                       X[1:,1:].flatten(),
                       Y[:-1,:-1].flatten(),
                       Y[1:,1:].flatten(),
                       C.flatten()):
        x = np.mean([x1, x2])
        y = np.mean([y1, y2])
        if not np.isnan(c):
            # text = ax.text(x, y, "{:.3f}".format(c), ha="center", va="center", color="w", rotation=45, size=5)
            text = ax.text(x, y, "{:.3f}".format(c), ha="center", va="center", color="w", rotation=0, size=7)

print('loading file '+file_name.replace("training", "testing"))
print('using models from '+out_dir)

df_test  = load_data(file_name.replace("training", "testing"), isEE=False, entrystop=200000)
X_test  = df_test[features_EB]
y_test  = df_test["target"]
xgtest  = xgb.DMatrix(X_test , label=y_test)

xgbo_default = xgb.Booster({'nthread':4}) #init model
#xgbo_default.load_model("Photon_perfectIC-highpt-EB/model_default/model.bin") # load data
xgbo_default.load_model(out_dir+"/model_default/model.bin")
xgbo_bo = xgb.Booster({'nthread':4}) #init model
#xgbo_bo.load_model("Photon_perfectIC-highpt-EB/model_optimized/model.bin") # load data
xgbo_bo.load_model(out_dir+"/model_optimized/model.bin") # load data

preds_default = xgbo_default.predict(xgtest)
preds_bo      = xgbo_bo.predict(xgtest)

# preds = 1.

# Etrue / Eraw after applying corrections
z_default = preds_default / (df_test['genEnergy']/df_test['rawEnergy'])
z_bo      = preds_bo / (df_test['genEnergy']/df_test['rawEnergy'])

bins = np.linspace(0.0, 2.0, 200)
bins_zoom = np.linspace(0.9, 1.1, 200)

plt.hist(df_test['rawEnergy']/df_test['genEnergy'], bins=bins, histtype='step', label="uncorrected")
plt.hist(z_default, bins=bins, histtype='step', label='corrected (default)')
plt.hist(z_bo, bins=bins, histtype='step', label='corrected (optimized)')
ax = plt.gca()
plt.title("Electrons in EB with perfectIC")
plt.xlabel("E measured / E gen")
plt.legend(loc="upper left")
plt.savefig(os.path.join(out_dir, "dist.png"))
ax.set_yscale("log", nonposy='clip')
plt.savefig(os.path.join(out_dir, "dist_log.png"))
plt.close()

plt.hist(df_test['rawEnergy']/df_test['genEnergy'], bins=bins_zoom, histtype='step', label="uncorrected")
plt.hist(z_default, bins=bins_zoom, histtype='step', label='corrected (default)')
plt.hist(z_bo, bins=bins_zoom, histtype='step', label='corrected (optimized)')
ax = plt.gca()
plt.title("Electrons in EB with perfectIC")
plt.xlabel("E measured / E gen")
plt.legend(loc="upper left")
plt.savefig(os.path.join(out_dir, "dist_zoom.png"))
ax.set_yscale("log", nonposy='clip')
plt.savefig(os.path.join(out_dir, "dist_log_zoom.png"))
plt.close()

print(np.mean(z_bo))
print(np.median(z_bo))
print(np.std(z_bo))

bins_eta = np.linspace(-1.5, 1.5, 7)
# bins_pt  = np.linspace(10, 300, 20)
bins_pt  = np.linspace(300, 1000, 20)


median, x_edges, y_edges, binnumber = stats.binned_statistic_2d(df_test["eta"], df_test["pt"], z_bo, bins=[bins_eta, bins_pt], statistic='median')
effrms, x_edges, y_edges, binnumber = stats.binned_statistic_2d(df_test["eta"], df_test["pt"], z_bo, bins=[bins_eta, bins_pt], statistic=rmseff)

X,Y = np.meshgrid(x_edges, y_edges)
plt.title("Median (corrected)")
plt.pcolor(X, Y, median.T)
plt.colorbar()
print_pcolor_labels(plt.gca(), X, Y, median.T)
plt.xlabel("eta")
plt.ylabel("pt")
plt.savefig(os.path.join(out_dir, "median.png"))
plt.close()

X,Y = np.meshgrid(x_edges, y_edges)
plt.title("Eff. Std Dev. (corrected)")
plt.pcolor(X, Y, effrms.T)
plt.colorbar()
print_pcolor_labels(plt.gca(), X, Y, effrms.T)
plt.xlabel("eta")
plt.ylabel("pt")
plt.savefig(os.path.join(out_dir, "effrms.png"))
plt.close()
