import numpy as np
from bayesian_optimization import BayesianOptimization
import xgboost as xgb
from scipy.special import logit
import pandas as pd
import time
from xgb_callbacks import callback_overtraining, callback_print_info, callback_timeout, early_stop
import os


def evaleffrms(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    x = np.sort(preds/labels, kind='mergesort')
    m = int(0.683*len(x)) + 1
    effrms = np.min(x[m:] - x[:-m])/2.0
    return 'effrms', effrms #+ 10*(max(np.median(preds/labels), np.median(labels/preds)) - 1)

def evalmedian(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'median', np.median(preds/labels)

def evalcustom(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    x = np.sort(preds/labels, kind='mergesort')
    m = int(0.683*len(x)) + 1
    effrms = np.min(x[m:] - x[:-m])/2.0
    return 'custom', effrms * (max(np.median(preds/labels), np.median(labels/preds)) - 1)

# The space of hyperparameters for the Bayesian optimization
hyperparams_ranges = {'min_child_weight': (1, 20),
                    'colsample_bytree': (0.1, 1),
                    'max_depth': (2, 15),
                    'subsample': (0.5, 1),
                    'gamma': (0, 10),
                    'reg_alpha': (0, 10),
                    'reg_lambda': (0, 10)}

# The default xgboost parameters
xgb_default = {'min_child_weight': 1,
               'colsample_bytree': 1,
               'max_depth': 6,
               'subsample': 1,
               'gamma': 0,
               'reg_alpha': 0,
               'reg_lambda': 1}

def format_params(params):
    """ Casts the hyperparameters to the required type and range.
    """
    p = dict(params)
    p['min_child_weight'] = p["min_child_weight"]
    p['colsample_bytree'] = max(min(p["colsample_bytree"], 1), 0)
    p['max_depth']        = int(p["max_depth"])
    p['subsample']        = max(min(p["subsample"], 1), 0)
    p['gamma']            = max(p["gamma"], 0)
    p['reg_alpha']        = max(p["reg_alpha"], 0)
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
                 early_stop_rounds = 10,
                 max_run_time      = 180000, # 50 h
                 train_time_factor = 5,
                 nthread           = 16,
                 regression        = False,
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

        if not os.path.exists(os.path.join(out_dir, "cv_results")):
            os.makedirs(os.path.join(out_dir, "cv_results"))

        self._random_state      = random_state
        self._num_rounds_max    = num_rounds_max
        self._early_stop_rounds = early_stop_rounds
        self._max_run_time      = max_run_time
        self._train_time_factor = train_time_factor

        self._start_time        = None
        self._max_training_time = None

        self.params_base = {
            'silent'      : 1,
            'verbose_eval': 0,
            'seed'        : self._random_state,
            'nthread'     : nthread,
            'objective'   : 'reg:linear',
            }

        if regression:
            # self._cv_cols =  ["train-rmse-mean", "train-rmse-std",
                              # "test-rmse-mean", "test-rmse-std"]
            self._cv_cols =  ["train-effrms-mean", "train-effrms-std",
                              "test-effrms-mean", "test-effrms-std"]

            # self.params_base['eval_metric'] = 'effrms'
        else:
            self._cv_cols =  ["train-auc-mean", "train-auc-std",
                              "test-auc-mean", "test-auc-std"]

            self.params_base['objective']   = 'binary;logitraw'
            self.params_base['eval_metric'] = 'auc'

        self._regression = regression

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

        self._params = {}

        self._cv_results = []
        self._cvi = 0

        #
        self._callback_status = []

        self._tried_default = False

        # self._summary = pd.DataFrame([
            # "callback", "colsample_bytree", "gamma", "max_depth",
            # "min_child_weight", "n_estimators", "reg_alpha", "reg_lambda",
            # "stage", "subsample"] + self._cv_cols)

    def evaluate_xgb(self, **hyperparameters):

        params = format_params(merge_two_dicts(self.params_base,
                                               hyperparameters))

        if len(self._bo.res["all"]["values"]) == 0:
            best_test_eval_metric = 0.0
        else:
            best_test_eval_metric = self._bo.res["all"]["values"][0]

        if self._max_training_time is None and not self._train_time_factor is None:
            training_start_time = time.time()

        if self._regression:
            callbacks= [callback_print_info(),
                        early_stop(self._early_stop_rounds, start_round=200, verbose=True, eval_idx=-2)]
            feval   = evaleffrms
        else:
            callbacks= [callback_print_info(),
                        early_stop(self._early_stop_rounds, verbose=True),
                        callback_overtraining(best_test_eval_metric, callback_status),
                        callback_timeout(self._max_training_time, best_test_eval_metric, callback_status)]
            feval   = None

        callback_status = {"status": 0}
        cv_result = xgb.cv(params, self._xgtrain,
                           num_boost_round=self._num_rounds_max,
                           nfold=self._nfold,
                           seed=self._random_state,
                           callbacks=callbacks,
                           feval=feval)

        cv_result.to_csv(os.path.join(self._out_dir, "cv_results/{0:04d}.csv".format(self._cvi)))
        self._cvi = self._cvi+1

        if self._max_training_time is None and not self._train_time_factor is None:
            self._max_training_time = self._train_time_factor * (time.time() - training_start_time)

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

        self._start_time = time.time()

        # Explore the default xgboost hyperparameters
        if not self._tried_default:
            self._bo.explore({k:[v] for k, v in xgb_default.items()}, eager=True)
            self._tried_default = True

        # Do the Bayesian optimization
        self._bo.maximize(init_points=init_points, n_iter=0, acq=acq)

        self._started_bo = True
        for i in range(n_iter):
            self._bo.maximize(init_points=0, n_iter=1, acq=acq)
            self.summary.to_csv(os.path.join(self._out_dir, "optimization.csv"))

            if not self._max_run_time is None and time.time() - self._start_time > self._max_run_time:
                print("Bayesian optimization timeout")
                break

        # Set up the parameters for the default training
        self._params["default"] = merge_two_dicts(self.params_base, xgb_default)
        self._params["default"]["n_estimators"] = self._early_stops[0]

        # Set up the parameters for the Bayesian-optimized training
        self._params["optimized"] = merge_two_dicts(self.params_base,
                                    format_params(self._bo.res["max"]["max_params"]))
        self._params["optimized"]["n_estimators"] = self._early_stops[np.argmax(self._bo.res["all"]["values"])]

    def fit(self, xgtrain, model="optimized"):
        print("Fitting with parameters")
        print(self._params[model])
        self._models[model] = xgb.train(self._params[model], xgtrain, self._params[model]["n_estimators"])

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

        for k in hyperparams_ranges.keys():
            data[k] = [res["params"][i][k] for i in range(n)]

        data["n_estimators"] = self._early_stops
        data["callback"] = self._callback_status

        return pd.DataFrame(data=data)


class XgboRegressor(XgboFitter):
    def __init__(self, out_dir,
                 random_state      = 2018,
                 num_rounds_max    = 3000,
                 early_stop_rounds = 10,
                 max_run_time      = 180000, # 50 h
                 train_time_factor = 5,
                 nthread           = 16,
            ):
        super(XgboRegressor, self).__init__(out_dir,
                                            random_state      = random_state,
                                            num_rounds_max    = num_rounds_max,
                                            early_stop_rounds = early_stop_rounds,
                                            max_run_time      = max_run_time, # 50 h
                                            train_time_factor = train_time_factor,
                                            nthread           = nthread,
                                            regression        = True)


class XgboClassifier(XgboFitter):
    def __init__(self, out_dir,
                 random_state      = 2018,
                 num_rounds_max    = 3000,
                 early_stop_rounds = 10,
                 max_run_time      = 180000, # 50 h
                 train_time_factor = 5,
                 nthread           = 16,
            ):
        super(XgboClassifier, self).__init__(out_dir,
                                             random_state      = random_state,
                                             num_rounds_max    = num_rounds_max,
                                             early_stop_rounds = early_stop_rounds,
                                             max_run_time      = max_run_time, # 50 h
                                             train_time_factor = train_time_factor,
                                             nthread           = nthread,
                                             regression        = False)
