import uproot
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy import stats
from xgbo import XgboRegressor
import os
import xgboost2tmva

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

    df = root_file['een_analyzer/ElectronTree'].pandas.df(branches, entrystop=entrystop).dropna()
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

# # Launched on polui04 with 5 50
# file_name = "/eos/cms/store/group/phys_egamma/EgammaRegression/94X/Electron/perfectIC-lowpt-EB-training.root"

# # Launched on polui03 with 5 1
# file_name = "/eos/cms/store/group/phys_egamma/EgammaRegression/94X/Electron/perfectIC-lowpt-EE-training.root"

# # Launched on polui01 with 0 1
# file_name = "/eos/cms/store/group/phys_egamma/EgammaRegression/94X/Electron/perfectIC-highpt-EE-training.root"

# Launched on polui03 with 5 1
file_name = "/eos/cms/store/group/phys_egamma/EgammaRegression/94X/Electron/perfectIC-highpt-EB-training.root"

isEE = '-EE-' in file_name

if isEE:
    features = features_EE
else:
    features = features_EB

tmp = file_name.split("/")
out_dir = tmp[-2] + "_" + tmp[-1].replace("-training.root", "")

df_train = load_data(file_name, isEE=isEE)
X_train = df_train[features]
y_train = df_train["target"]
xgtrain = xgb.DMatrix(X_train, label=y_train)

# Create the XgboRegressor
xgbo_reg = XgboRegressor(out_dir, early_stop_rounds=100)

xgbo_reg.optimize(xgtrain, init_points=5, n_iter=1, acq='ei')
# xgbo_reg.optimize(xgtrain, init_points=1, n_iter=0, acq='ei')
print(xgbo_reg.summary)

df_test  = load_data(file_name.replace("training", "testing"), isEE=isEE)
X_test  = df_test[features]
y_test  = df_test["target"]
xgtest  = xgb.DMatrix(X_test , label=y_test)

xgbo_reg.fit(xgtrain, model="default")
xgbo_reg.fit(xgtrain, model="optimized")

preds_default = xgbo_reg.predict(xgtest, model="default")
preds_bo      = xgbo_reg.predict(xgtest, model="optimized")

# preds = 1.

# Etrue / Eraw after applying corrections
z_default = preds_default / (df_test['genEnergy']/df_test['rawEnergy'])
z_bo      = preds_bo / (df_test['genEnergy']/df_test['rawEnergy'])


print("Saving weight files...")
xgbo_reg._models["default"].dump_model(os.path.join(out_dir, 'model_default.txt'))
tmvafile = os.path.join(out_dir, "weights_default.xml")
xgboost2tmva.convert_model(xgbo_reg._models["default"].get_dump(),
                           input_variables = list(zip(features_EB, len(features_EB)*['F'])),
                           output_xml = tmvafile)
os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
os.system("mv {0} {0}.bak".format(tmvafile))
os.system("mv {0}.tmp {0}".format(tmvafile))
os.system("cd "+ out_dir + " && gzip -f weights.xml")

xgbo_reg._models["optimized"].dump_model(os.path.join(out_dir, 'model_optimized.txt'))
tmvafile = os.path.join(out_dir, "weights_optimized.xml")
xgboost2tmva.convert_model(xgbo_reg._models["optimized"].get_dump(),
                           input_variables = list(zip(features_EB, len(features_EB)*['F'])),
                           output_xml = tmvafile)
os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
os.system("mv {0} {0}.bak".format(tmvafile))
os.system("mv {0}.tmp {0}".format(tmvafile))
os.system("cd "+ out_dir + " && gzip -f weights.xml")

"""
bins = np.linspace(0.0, 2.0, 200)

plt.hist(df_test['rawEnergy']/df_test['genEnergy'], bins=bins, histtype='step', label="uncorrected")
plt.hist(z_default, bins=bins, histtype='step', label='corrected')
plt.hist(z_bo, bins=bins, histtype='step', label='corrected')
ax = plt.gca()
ax.set_yscale("log", nonposy='clip')
plt.xlabel("E measured / E gen")
plt.legend(loc="upper left")
plt.savefig(os.path.join(out_dir, "dist.png"))
plt.close()

print(np.mean(z_bo))
print(np.median(z_bo))
print(np.std(z_bo))

# bins_eta = np.array([-2.5, -2.0, -1.5, -0.8, 0, 0.8, 1.5, 2.0, 2.5])
bins_eta = np.linspace(-1.5, 1.5, 7)
bins_pt  = np.linspace(10, 300, 20)


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
"""
