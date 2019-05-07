import os
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

from .bayesian_optimization import BayesianOptimization
from .xgb_callbacks import callback_overtraining, early_stop
from .xgboost2tmva import convert_model

import warnings

# Effective RMS evaluation function for xgboost
def evaleffrms(preds, dtrain, c=0.683):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    x = np.sort(preds/labels, kind='mergesort')
    m = int(c*len(x)) + 1
    effrms = np.min(x[m:] - x[:-m])/2.0
    return 'effrms', effrms #+ 10*(max(np.median(preds/labels), np.median(labels/preds)) - 1)

# The space of hyperparameters for the Bayesian optimization
#hyperparams_ranges = {'min_child_weight': (1, 30),
#                    'colsample_bytree': (0.1, 1),
#                    'max_depth': (2, 20),
#                    'subsample': (0.5, 1),
#                    'gamma': (0, 20),
#                    'reg_alpha': (0, 10),
#                    'reg_lambda': (0, 20)}

# The default xgboost parameters
#xgb_default = {'min_child_weight': 1,
#               'colsample_bytree': 1,
#               'max_depth': 6,
#               'subsample': 1,
#               'gamma': 0,
#               'reg_alpha': 0,
#               'reg_lambda': 1}

def format_params(params):
    """ Casts the hyperparameters to the required type and range.
    """
    p = dict(params)
    p['min_child_weight'] = p["min_child_weight"]
    p['colsample_bytree'] = max(min(p["colsample_bytree"], 1), 0)
    p['max_depth']        = int(p["max_depth"])
#    p['subsample']        = max(min(p["subsample"], 1), 0)
    p['gamma']            = max(p["gamma"], 0)
#    p['reg_alpha']        = max(p["reg_alpha"], 0)
    p['reg_lambda']       = max(p["reg_lambda"], 0)
    return p

def merge_two_dicts(x, y):
    """ Merge two dictionaries.

    Writing such a function is necessary in Python 2.

    In Python 3, one can just do:
        d_merged = {**d1, **d2}.
    """
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

class XgboFitter(object):
    """Fits a xgboost classifier/regressor with Bayesian-optimized hyperparameters.

    Public attributes:

    Private attributes:
        _random_state (int): seed for random number generation
    """

    def __init__(self, out_dir,
                 random_state      = 2018,
                 num_rounds_max    = 3000,
                 num_rounds_min    = 0,
                 early_stop_rounds = 100,
                 nthread           = 16,
                 regression        = False,
                 useEffSigma       =True
            ):
        """The __init__ method for XgboFitter class.

        Args:
            data (pandas.DataFrame): The  data frame containing the features
                                     and target.
            X_cols (:obj:`list` of :obj:`str`) : Names of the feature columns.
            y_col (str) : Name of the colum containing the target of the binary
                          classification. This column has to contain zeros and
                          ones.
        """
        self._out_dir = out_dir
        pkl_file = open(out_dir+'/param_range.pkl', 'rb')
        global hyperparams_ranges
        hyperparams_ranges= pickle.load(pkl_file)
        pkl_file.close()
        pkl_file = open(out_dir+'/param_default.pkl', 'rb')
        global xgb_default
        xgb_default= pickle.load(pkl_file)
        pkl_file.close()
        if not os.path.exists(os.path.join(out_dir, "cv_results")):
            os.makedirs(os.path.join(out_dir, "cv_results"))

        self._random_state      = random_state
        self._num_rounds_max    = num_rounds_max
        self._num_rounds_min    = num_rounds_min
        self._early_stop_rounds = early_stop_rounds

        self.params_base = {
            'silent'      : 1,
            'verbose_eval': 0,
            'seed'        : self._random_state,
            'nthread'     : nthread,
            'objective'   : 'reg:linear',
            }
        
        if regression:
            xgb_default['base_score']=1#for regression the base_score should be 1, not 0.5. If enough iteration this will not matter much
            if useEffSigma:
                self._cv_cols =  ["train-effrms-mean", "train-effrms-std",
                                  "test-effrms-mean", "test-effrms-std"]
            else:
                self._cv_cols =  ["train-rmse-mean", "train-rmse-std",
                                  "test-rmse-mean", "test-rmse-std"]
        else:
            self._cv_cols =  ["train-auc-mean", "train-auc-std",
                              "test-auc-mean", "test-auc-std"]

            self.params_base['objective']   = 'binary:logitraw'
            self.params_base['eval_metric'] = 'auc'

        self._regression = regression
        self._useEffSigma = useEffSigma

        # Increment the random state by the number of previously done
        # experiments so we don't use the same numbers twice
        summary_file = os.path.join(out_dir, "summary.csv")
        if os.path.isfile(summary_file):
            df = pd.read_csv(summary_file)
            self._random_state = self._random_state + len(df)

        # Set up the Bayesian optimization
        self._bo = BayesianOptimization(self.evaluate_xgb,
                                        hyperparams_ranges,
                                        random_state=self._random_state)

        # This list will memorize the number of rounds that each step in the
        # Bayesian optimization was trained for before early stopping gets
        # triggered. This way, we can train our final classifier with the
        # correct n_estimators matching to the optimal hyperparameters.
        self._early_stops = []

        # This dictionary will hold the xgboost models created when running
        # this training class.
        self._models = {}

        self._cv_results = []
        self._cvi = 0

        #
        self._callback_status = []

        self._tried_default = False

        # Load the summary file if it already exists in the out_dir
        if os.path.isfile(summary_file):
            self._load_data()


    def _load_data(self):

        summary_file = os.path.join(self._out_dir, "summary.csv")

        df = pd.read_csv(summary_file)

        print("Found results of {} optimization rounds in ouptut directory, loading...".format(len(df)))

        self._early_stops     += list(df.n_estimators.values)
        self._callback_status += list(df.callback.values)

        self._tried_default = True

        # Load the cross validation results
        for i in range(len(df)):
            cv_file = os.path.join(self._out_dir, "cv_results/{0:04d}.csv".format(i))
            self._cv_results.append(pd.read_csv(cv_file))
        self._cvi = len(df)

        # Load the optimization results so far into the Bayesian optimization object
        eval_col = self._cv_cols[2]

        if self._regression:
            idx_max =  df[eval_col].idxmin()
            max_val = -df[eval_col].min()
        else:
            idx_max = df[eval_col].idxmax()
            max_val = df[eval_col].max()

        self._bo.res["max"] = {'max_val' : max_val,
                               'max_params' : df.loc[idx_max, list(hyperparams_ranges)].to_dict()}

        for idx in df.index:
            value = df.loc[idx, eval_col]
            if self._regression:
                value = -value
            self._bo.res["all"]["values"].append(value)
            self._bo.res["all"]["params"].append(df.loc[idx, list(hyperparams_ranges)].to_dict())

        if self._regression:
            df["target"] = -df[eval_col]
        else:
            df["target"] = df[eval_col]

        self._bo.initialize(df[["target"] + list(hyperparams_ranges)])

    def evaluate_xgb(self, **hyperparameters):

        params = format_params(merge_two_dicts(self.params_base,
                                               hyperparameters))

        if len(self._bo.res["all"]["values"]) == 0:
            best_test_eval_metric = -9999999.
        else:
            self.summary.to_csv(os.path.join(self._out_dir, "summary.csv"))
            best_test_eval_metric = self._bo.res["all"]["values"][0]

        feval     = None
        callback_status = {"status": 0}

        if self._regression:
            if self._useEffSigma:
                callbacks = [early_stop(self._early_stop_rounds, start_round=self._num_rounds_min, verbose=True, eval_idx=-2)]
                feval     = evaleffrms
            else:
                callbacks = [early_stop(self._early_stop_rounds, start_round=self._num_rounds_min, verbose=True),
                             callback_overtraining(best_test_eval_metric, callback_status)]
        else:
            callbacks = [early_stop(self._early_stop_rounds, start_round=self._num_rounds_min, verbose=True),
                         callback_overtraining(best_test_eval_metric, callback_status)]

        cv_result = xgb.cv(params, self._xgtrain,
                           num_boost_round=self._num_rounds_max,
                           nfold=self._nfold,
                           seed=self._random_state,
                           callbacks=callbacks,
                           verbose_eval=10,
                           feval=feval)

        cv_result.to_csv(os.path.join(self._out_dir, "cv_results/{0:04d}.csv".format(self._cvi)))
        self._cvi = self._cvi+1

        self._early_stops.append(len(cv_result))

        self._cv_results.append(cv_result)
        self._callback_status.append(callback_status['status'])

        if self._regression:
            return -cv_result[self._cv_cols[2]].values[-1]
        else:
            return cv_result[self._cv_cols[2]].values[-1]

    def optimize(self, xgtrain, init_points=3, n_iter=3, nfold=3, acq="ei"):

        self._nfold       = nfold

        # Save data in xgboosts DMatrix format so the encoding doesn't have to
        # be repeated at every step of the Bayesian optimization.
        self._xgtrain = xgtrain

        # Explore the default xgboost hyperparameters
        if not self._tried_default:
            self._bo.explore({k:[v] for k, v in xgb_default.items()}, eager=True)
            self._tried_default = True

        # Do the Bayesian optimization
        self._bo.maximize(init_points=init_points, n_iter=0, acq=acq)

        self._started_bo = True
        for i in range(n_iter):
            self._bo.maximize(init_points=0, n_iter=1, acq=acq)

            # Save summary after each step so we can interrupt at any time
            self.summary.to_csv(os.path.join(self._out_dir, "summary.csv"))

        # Final save of the summary
        self.summary.to_csv(os.path.join(self._out_dir, "summary.csv"))

    def fit(self, xgtrain, model="optimized"):

        if model == "default":
            # Set up the parameters for the default training
            params = merge_two_dicts(self.params_base, xgb_default)
            params["n_estimators"] = self._early_stops[0]

        if model == "optimized":
            # Set up the parameters for the Bayesian-optimized training
            params = merge_two_dicts(self.params_base,
                     format_params(self._bo.res["max"]["max_params"]))
            params["n_estimators"] = self._early_stops[np.argmax(self._bo.res["all"]["values"])]

        self._models[model] = xgb.train(params, xgtrain, params["n_estimators"], verbose_eval=10)

    def predict(self, xgtest, model="optimized"):
        return self._models[model].predict(xgtest)

    @property
    def summary(self):
        res = dict(self._bo.res["all"])

        n = len(res["params"])
        for i in range(n):
            res["params"][i] = format_params(res["params"][i])

        data = {}

        for name in self._cv_cols:
            data[name] = [cvr[name].values[-1] for cvr in self._cv_results]

        for k, v in hyperparams_ranges.items():
            data[k] = [res["params"][i][k] for i in range(n)]

        data["n_estimators"] = self._early_stops
        data["callback"] = self._callback_status

        return pd.DataFrame(data=data)

    def save_model(self, feature_names, model="optimized"):
        """Save model from booster to binary, text and XML.
        """
        print("Saving model")
        model_dir = os.path.join(self._out_dir, "model_" + model)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save text dump
        self._models[model].dump_model(os.path.join(model_dir, 'dump.raw.txt'))

        # Save in binary format
        self._models[model].save_model(os.path.join(model_dir, 'model.bin'))

        # Convert to TMVA or GBRForest compatible weights file
        tmvafile = os.path.join(model_dir, "weights.xml")
        try:
            convert_model(self._models[model].get_dump(),
                          input_variables = list(zip(feature_names, len(feature_names)*['F'])),
                          output_xml = tmvafile)
            os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
            os.system("mv {0} {0}.bak".format(tmvafile))
            os.system("mv {0}.tmp {0}".format(tmvafile))
            os.system("gzip -f {0}".format(tmvafile))
            os.system("mv {0}.bak {0}".format(tmvafile))
        except:
            warnings.warn("Warning:\nSaving model in TMVA XML format failed.\nDon't worry now, you can still convert the xgboost model later.")

class XgboRegressor(XgboFitter):
    def __init__(self, out_dir,
                 random_state      = 2018,
                 num_rounds_max    = 3000,
                 num_rounds_min    = 0,
                 early_stop_rounds = 100,
                 nthread           = 16,
            ):
        super(XgboRegressor, self).__init__(out_dir,
                                            random_state      = random_state,
                                            num_rounds_max    = num_rounds_max,
                                            num_rounds_min    = num_rounds_min,
                                            early_stop_rounds = early_stop_rounds,
                                            nthread           = nthread,
                                            regression        = True)


class XgboClassifier(XgboFitter):
    def __init__(self, out_dir,
                 random_state      = 2018,
                 num_rounds_max    = 3000,
                 num_rounds_min    = 0,
                 early_stop_rounds = 100,
                 nthread           = 16,
            ):
        super(XgboClassifier, self).__init__(out_dir,
                                             random_state      = random_state,
                                             num_rounds_max    = num_rounds_max,
                                             num_rounds_min    = num_rounds_min,
                                             early_stop_rounds = early_stop_rounds,
                                             nthread           = nthread,
                                             regression        = False)
