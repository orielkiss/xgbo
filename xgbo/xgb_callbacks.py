import time
import xgboost as xgb
from xgboost import rabit

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
        m = len(env.evaluation_result_list)
        train_auc = env.evaluation_result_list[0][1]
        test_auc  = env.evaluation_result_list[m/2][1]

        if n % n_skip == 0:
            print("[{0:4d}]\ttrain-eval:{1:.6f}\ttest-eval:{2:.6f}".format(n, train_auc, test_auc))

    return callback

def _fmt_metric(value, show_stdv=True):
    """format metric string"""
    if len(value) == 2:
        return '%s:%g' % (value[0], value[1])
    elif len(value) == 3:
        if show_stdv:
            return '%s:%g+%g' % (value[0], value[1], value[2])
        else:
            return '%s:%g' % (value[0], value[1])
    else:
        raise ValueError("wrong metric value")

# Modification of the official early_stop callback to only trigger it from the nth round on
def early_stop(stopping_rounds, start_round=0, maximize=False, verbose=True, eval_idx=-1):
    """Create a callback that activates early stoppping.
    Validation error needs to decrease at least
    every **stopping_rounds** round(s) to continue training.
    Requires at least one item in **evals**.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
    (Use ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree``
    and/or ``num_class`` appears in the parameters)
    Parameters
    ----------
    stopp_rounds : int
       The stopping rounds before the trend occur.
    maximize : bool
        Whether to maximize evaluation metric.
    verbose : optional, bool
        Whether to print message about early stopping information.
    Returns
    -------
    callback : function
        The requested callback function.
    """
    state = {}

    def init(env):
        """internal function"""
        bst = env.model

        if len(env.evaluation_result_list) == 0:
            raise ValueError('For early stopping you need at least one set in evals.')
        if len(env.evaluation_result_list) > 1 and verbose:
            msg = ("Multiple eval metrics have been passed: "
                   "'{0}' will be used for early stopping.\n\n")
            rabit.tracker_print(msg.format(env.evaluation_result_list[eval_idx][0]))
        maximize_metrics = ('auc', 'map', 'ndcg')
        maximize_at_n_metrics = ('auc@', 'map@', 'ndcg@')
        maximize_score = maximize
        metric_label = env.evaluation_result_list[eval_idx][0]
        metric = metric_label.split('-', 1)[-1]

        if any(metric.startswith(x) for x in maximize_at_n_metrics):
            maximize_score = True

        if any(metric.split(":")[0] == x for x in maximize_metrics):
            maximize_score = True

        if verbose and env.rank == 0:
            msg = "Will train until {} hasn't improved in {} rounds.\n"
            rabit.tracker_print(msg.format(metric_label, stopping_rounds))

        state['maximize_score'] = maximize_score
        state['best_iteration'] = 0
        if maximize_score:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if env.iteration < start_round:
            return

        score = env.evaluation_result_list[eval_idx][1]
        if len(state) == 0:
            init(env)
        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d]\t%s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose and env.rank == 0:
                msg = "Stopping. Best iteration:\n{}\n\n"
                rabit.tracker_print(msg.format(best_msg))
            raise xgb.core.EarlyStopException(best_iteration)

    return callback
