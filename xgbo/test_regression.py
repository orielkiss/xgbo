import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from xgbo import XgboRegressor
from sklearn.model_selection import train_test_split

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

# Split in testing and training subsamples
X_train, X_test, y_train, y_test = \
        train_test_split(df[features], df[target], random_state=99, test_size=0.25)

# Set up the DMatrix for xgboost
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest  = xgb.DMatrix(X_test , label=y_test)

# Create the XgboRegressor
xgbo_reg = XgboRegressor()

xgbo_reg.optimize(xgtrain, init_points=1, n_iter=1, acq='ei')
print(xgbo_reg.summary)
xgbo_reg.fit(xgtrain)

preds = xgbo_reg.predict(xgtest, model="optimized")


plt.hist(df[target], bins=np.linspace(0, 100, 200), histtype='step')
plt.hist(preds, bins=np.linspace(0, 100, 200), histtype='step')
# plt.show()
