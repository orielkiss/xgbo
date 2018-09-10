import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from xgbo import XgboRegressor

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

# Create the XgboRegressor
xgbo_reg = XgboRegressor()

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

plt.hist(preds, bins=np.linspace(0, 100, 200), histtype='step')
plt.show()

plt.hist((preds - df[target]), bins=200)
plt.show()
