import pandas as pd
from xgb_quantile_loss import xgb_quantile_obj, xgb_quantile_eval
import xgboost as xgb
import numpy as np

df = pd.read_hdf("../res/electron_data.root")

# In this example, let us try to regress the pt of true electrons in the EB based on the other information

# Select true electrons in EB only
df = df.query("matchedToGenEle == 1 & scl_eta < 1.5")

# The list of features we use
features = [
             "ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi",
             "ele_oldcircularity", "ele_oldr9", "ele_scletawidth",
             "ele_sclphiwidth", "ele_oldhe", "ele_kfhits", "ele_kfchi2",
             "ele_gsfchi2", "ele_fbrem", "ele_gsfhits",
             "ele_expected_inner_hits", "ele_conversionVertexFitProbability",
             "ele_ep", "ele_eelepout", "ele_IoEmIop", "ele_deltaetain",
             "ele_deltaphiin", "ele_deltaetaseed", "rho",
             "ele_pfPhotonIso", "ele_pfChargedHadIso", "ele_pfNeutralHadIso"]

# What we want to regress
target = "ele_pt"

print(df.head())

# Set up the DMatrix for xgboost
train = xgb.DMatrix(df[features], label=df[target])

# The default xgboost parameters
xgb_default = {'min_child_weight': 1,
               'colsample_bytree': 1,
               'max_depth': 6,
               'subsample': 1,
               'gamma': 0,
               'reg_alpha': 0,
               'reg_lambda': 1}

bst = xgb.train(xgb_default, train, 100)
preds = bst.predict(train)

"""
bst68 = xgb.train(xgb_default, train, 1000,
                obj=lambda preds, dmatrix : xgb_quantile_obj(preds, dmatrix, quantile=0.68))
                #, feval=xgb_quantile_eval)
preds68 = bst68.predict(train)

bst32 = xgb.train(xgb_default, train, 1000,
                obj=lambda preds, dmatrix : xgb_quantile_obj(preds, dmatrix, quantile=0.32))
                #, feval=xgb_quantile_eval)
preds32 = bst32.predict(train)

stddev = (preds68 - preds32)/2

import matplotlib.pyplot as plt
plt.hist(preds32, bins=np.linspace(0, 100, 200), histtype='step')
plt.hist(preds50, bins=np.linspace(0, 100, 200), histtype='step')
plt.hist(preds68, bins=np.linspace(0, 100, 200), histtype='step')
plt.show()

plt.hist((preds50 - df[target]), bins=200)
plt.show()

plt.hist((preds50 - df[target])/stddev, bins=np.linspace(-10, 10, 200))
plt.show()
"""
