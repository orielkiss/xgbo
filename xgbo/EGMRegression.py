import uproot
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy import stats
from xgbo import XgboRegressor

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

def load_data(file_name, entrystop=50000):

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

    branches = branches_EB + ["pt", "eta"]

    # print(len(root_file['een_analyzer/ElectronTree'].pandas.df(["rawEnergy"]).dropna()))
    df = root_file['een_analyzer/ElectronTree'].pandas.df(branches, entrystop=entrystop).dropna()

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

    # The target
    # df.eval("target = genEnergy / ( rawEnergy + preshowerEnergy )", inplace=True)
    df.eval("target = genEnergy / rawEnergy", inplace=True)

    return df

df_train = load_data("/eos/cms/store/group/phys_egamma/EgammaRegression/94X/Electron/perfectIC-lowpt-EB-training.root")
df_test  = load_data("/eos/cms/store/group/phys_egamma/EgammaRegression/94X/Electron/perfectIC-lowpt-EB-testing.root")

# The features
features_EB = [ 'rawEnergy', 'etaWidth', 'phiWidth', 'rhoValue',
        'full5x5_sigmaIetaIeta', 'full5x5_sigmaIetaIphi',
        'full5x5_sigmaIphiIphi', 'clusertRawEnergyOverE5x5', 'w3x3OverE5x5',
        'eMaxOverE5x5', 'e2ndOverE5x5', 'eTopOverE5x5', 'eBottomOverE5x5',
        'eLeftOverE5x5', 'eRightOverE5x5', 'e2x5MaxOverE5x5',
        'e2x5TopOverE5x5', 'e2x5BottomOverE5x5', 'e2x5LeftOverE5x5',
        'e2x5RightOverE5x5', 'iEtaSeed', 'iPhiSeed', 'iEtaMod5', 'iPhiMod2',
        'iEtaMod20', 'iPhiMod20']

# # EE
features_EE = [ 'rawEnergy', 'etaWidth', 'phiWidth', 'rhoValue',
        'full5x5_sigmaIetaIeta', 'full5x5_sigmaIetaIphi',
        'full5x5_sigmaIphiIphi', 'clusertRawEnergyOverE5x5', 'w3x3OverE5x5',
        'eMaxOverE5x5', 'e2ndOverE5x5', 'eTopOverE5x5', 'eBottomOverE5x5',
        'eLeftOverE5x5', 'eRightOverE5x5', 'e2x5MaxOverE5x5',
        'e2x5TopOverE5x5', 'e2x5BottomOverE5x5', 'e2x5LeftOverE5x5',
        'e2x5RightOverE5x5', 'iXSeed', 'iYSeed', 'preshowerEnergy/rawEnergy']

print(df_train[features_EB].tail())


X_train = df_train[features_EB]
X_test  = df_test[features_EB]

y_train = df_train["target"]
y_test  = df_test["target"]

# Set up the DMatrix for xgboost
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest  = xgb.DMatrix(X_test , label=y_test)

# Create the XgboRegressor
xgbo_reg = XgboRegressor(early_stop_rounds=20)

xgbo_reg.optimize(xgtrain, init_points=20, n_iter=10, acq='ucb')
xgbo_reg.optimize(xgtrain, init_points=20, n_iter=10, acq='ei')
print(xgbo_reg.summary)
xgbo_reg.fit(xgtrain, model="default")
xgbo_reg.fit(xgtrain, model="optimized")

preds_default = xgbo_reg.predict(xgtest, model="default")
preds_bo      = xgbo_reg.predict(xgtest, model="optimized")

# preds = 1.

# Etrue / Eraw after applying corrections
z_default = preds_default / (df_test['genEnergy']/df_test['rawEnergy'])
z_bo      = preds_bo / (df_test['genEnergy']/df_test['rawEnergy'])

# bins = np.linspace(0.7, 1.3, 200)
bins = np.linspace(0.0, 2.0, 200)

# plt.hist(preds_default, bins=bins, histtype='step', label='corrected')
# plt.hist(preds_bo, bins=bins, histtype='step', label='corrected')
# plt.savefig("fig1.png")
# plt.close()

# plt.hist(preds_default - df_test['genEnergy']/df_test['rawEnergy'], bins=bins, histtype='step', label='corrected')
# plt.hist(preds_bo - df_test['genEnergy']/df_test['rawEnergy'], bins=bins, histtype='step', label='corrected')
# plt.savefig("fig2.png")
# plt.close()

plt.hist(df_test['rawEnergy']/df_test['genEnergy'], bins=bins, histtype='step', label="uncorrected")
plt.hist(z_default, bins=bins, histtype='step', label='corrected')
plt.hist(z_bo, bins=bins, histtype='step', label='corrected')
ax = plt.gca()
ax.set_yscale("log", nonposy='clip')
plt.xlabel("E measured / E gen")
plt.legend(loc="upper left")
plt.savefig("dist.png")
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
plt.savefig("median.png")
plt.close()

X,Y = np.meshgrid(x_edges, y_edges)
plt.title("Eff. Std Dev. (corrected)")
plt.pcolor(X, Y, effrms.T)
plt.colorbar()
print_pcolor_labels(plt.gca(), X, Y, effrms.T)
plt.xlabel("eta")
plt.ylabel("pt")
plt.savefig("effrms.png")
plt.close()
