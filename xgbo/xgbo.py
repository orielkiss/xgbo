from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from bayes_opt import BayesianOptimization
import xgboost as xgb
from scipy.special import logit
import pandas as pd
import time

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

def callback_overtraining(best_test_auc, callback_status):

    def callback(env):
        train_auc = env.evaluation_result_list[0][1]
        test_auc = env.evaluation_result_list[1][1]

        if train_auc < best_test_auc:
            return

        if train_auc - test_auc > 1 - best_test_auc:
            print("We have an overtraining problem! Stop boosting.")
            callback_status["status"] = 2
            raise xgb.core.EarlyStopException(env.iteration)

    return callback

def callback_timeout(max_time, best_test_auc, callback_status, n_fit=10):

    start_time = time.time()

    last_n_times = []
    last_n_test_auc = []

    status = {'counter': 0}

    def callback(env):

        if max_time == None:
            return

        run_time = time.time() - start_time

        if run_time > max_time:
            callback_status["status"] = 3
            raise xgb.core.EarlyStopException(env.iteration)
            print("Xgboost training took too long. Stop boosting.")
            raise xgb.core.EarlyStopException(env.iteration)

        last_n_test_auc.append(env.evaluation_result_list[1][1])
        if len(last_n_test_auc) > n_fit:
            del last_n_test_auc[0]

        last_n_times.append(run_time)
        if len(last_n_times) > n_fit:
            del last_n_times[0]

        if len(last_n_test_auc) < n_fit:
            return

        poly = np.polyfit(last_n_times, last_n_test_auc, deg=1)
        guessed_test_auc_at_max_time  = np.polyval(poly, max_time)

        if guessed_test_auc_at_max_time < best_test_auc and best_test_auc > 0.0:
            status['counter'] = status['counter'] + 1
        else:
            status['counter'] = 0

        if status['counter'] == n_fit:
            callback_status["status"] = 2
            raise xgb.core.EarlyStopException(env.iteration)
            print("Test AUC does not converge well. Stop boosting.")
            raise xgb.core.EarlyStopException(env.iteration)

    return callback

def callback_print_info(n_skip=10):

    def callback(env):
        n = env.iteration
        train_auc = env.evaluation_result_list[0][1]
        test_auc  = env.evaluation_result_list[1][1]

        if n % n_skip == 0:
            print("[{0:4d}]\ttrain-auc:{1:.6f}\ttest-auc:{2:.6f}".format(n, train_auc, test_auc))

    return callback

class XgbBoTrainer:
    """Trains a xgboost classifier with Bayesian-optimized hyperparameters.

    Public attributes:

    Private attributes:
        _random_state (int): seed for random number generation
    """

    def __init__(self, data, X_cols, y_col,
                 random_state      = 2018,
                 num_rounds_max    = 3000,
                 early_stop_rounds = 10,
                 nfold             = 3,
                 init_points       = 5,
                 n_iter            = 50,
                 max_run_time      = 180000, # 50 h
                 train_time_factor = 5,
                 test_size         = 0.25,
                 max_n_per_class   = None,
                 nthread           = 16,
            ):
        """The __init__ method for XgbBoTrainer class.

        Args:
            data (pandas.DataFrame): The  data frame containing the features
                                     and target.
            X_cols (:obj:`list` of :obj:`str`) : Names of the feature columns.
            y_col (str) : Name of the colum containing the target of the binary
                          classification. This column has to contain zeros and
                          ones.
        """
        self._random_state      = random_state
        self._num_rounds_max    = num_rounds_max
        self._early_stop_rounds = early_stop_rounds
        self._nfold             = nfold
        self._init_points       = init_points
        self._n_iter            = n_iter
        self._max_run_time      = max_run_time
        self._train_time_factor = train_time_factor

        self._start_time        = None
        self._max_training_time = None

        self.params_base = {
            'silent'      : 1,
            'eval_metric' : 'auc',
            'verbose_eval': 0,
            'seed'        : self._random_state,
            'nthread'     : nthread,
            'objective'   : 'binary:logitraw',
            }

        # Entries from the class with more entries are discarded. This is because
        # classifier performance is usually bottlenecked by the size of the
        # dataset for the class with fewer entries. Having one class with extra
        # statistics usually just adds computing time.
        n_per_class = min(min(data[y_col].value_counts()), max_n_per_class)

        # The number of entries per class can be limited by a parameter in case
        # the dataset is just too large for this algorithm to run in a
        # reasonable time.
        if not max_n_per_class is None:
            n_per_class = min(n_per_class, max_n_per_class)

        data = pd.concat([data[data[y_col] == 0].head(n_per_class),
                          data[data[y_col] == 1].head(n_per_class)])

        # Split in testing and training subsamples
        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(data[X_cols],
                                 data[y_col],
                                 random_state=self._random_state,
                                 test_size=test_size)

        print(self.X_train.columns)

        # Save data in xgboosts DMatrix format so the encoding doesn't have to
        # be repeated at every step of the Bayesian optimization.
        self._xgtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self._xgtest  = xgb.DMatrix(self.X_test, label=self.y_test)

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
        self.models = {}

        self.cv_results = []

        #
        self._callback_status = []

    def evaluate_xgb(self, **hyperparameters):

        params = format_params(merge_two_dicts(self.params_base,
                                               hyperparameters))

        if len(self._bo.res["all"]["values"]) == 0:
            best_test_auc = 0.0
        else:
            best_test_auc = self._bo.res["all"]["values"][0]

        if self._max_training_time is None and not self._train_time_factor is None:
            training_start_time = time.time()

        callback_status = {"status": 0}
        cv_result = xgb.cv(params, self._xgtrain,
                           num_boost_round=self._num_rounds_max,
                           nfold=self._nfold,
                           seed=self._random_state,
                           callbacks=[
                               callback_print_info(),
                               xgb.callback.early_stop(self._early_stop_rounds, verbose=True),
                               callback_overtraining(best_test_auc, callback_status),
                               callback_timeout(self._max_training_time, best_test_auc, callback_status),
                           ])

        if self._max_training_time is None and not self._train_time_factor is None:
            self._max_training_time = self._train_time_factor * (time.time() - training_start_time)

        self._early_stops.append(len(cv_result))

        self.cv_results.append(cv_result)
        self._callback_status.append(callback_status['status'])

        return cv_result['test-auc-mean'].values[-1]

    def run(self):

        self._start_time = time.time()

        # Explore the default xgboost hyperparameters
        self._bo.explore({k:[v] for k, v in xgb_default.items()}, eager=True)

        # Do the Bayesian optimization
        self._bo.maximize(init_points=self._init_points, n_iter=0, acq='ei')

        self._started_bo = True
        for i in range(self._n_iter):
            self._bo.maximize(init_points=0, n_iter=1, acq='ei')

            if not self._max_run_time is None and time.time() - self._start_time > self._max_run_time:
                print("Bayesian optimization timeout")
                break

        # Set up the parameters for the default training
        params_default = merge_two_dicts(self.params_base, xgb_default)
        params_default["n_estimators"] = self._early_stops[0]

        # Set up the parameters for the Bayesian-optimized training
        params_bo = merge_two_dicts(self.params_base,
                                    format_params(self._bo.res["max"]["max_params"]))
        params_bo["n_estimators"] = self._early_stops[np.argmax(self._bo.res["all"]["values"])]

        # Fit default model to test sample
        self.models["default"] = xgb.XGBClassifier(**params_default)
        self.models["default"].fit(self.X_train, self.y_train)

        # Fit Bayesian-optimized model to test sample
        self.models["bo"] = xgb.XGBClassifier(**params_bo)
        self.models["bo"].fit(self.X_train,self.y_train)

        return self._bo.res

    def get_score(self, model_name):
        return self.models[model_name]._Booster.predict(self._xgtest)

    def get_roc(self, model_name):
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, self.get_score(model_name))

        return fpr, tpr, thresholds

    def get_results_df(self):
        res = dict(self._bo.res["all"])

        n = len(res["params"])
        for i in range(n):
            res["params"][i] = format_params(res["params"][i])

        n_iter_eff = n - 1 - self._init_points

        data = {}

        data["stage"] = [0] + [1] * self._init_points + [2] * n_iter_eff

        for name in ["train-auc-mean", "train-auc-std",
                     "test-auc-mean", "test-auc-std"]:
            data[name] = [cvr[name].values[-1] for cvr in self.cv_results]

        for k in hyperparams_ranges.keys():
            data[k] = [res["params"][i][k] for i in range(n)]

        data["n_estimators"] = self._early_stops
        data["callback"] = self._callback_status

        return pd.DataFrame(data=data)
