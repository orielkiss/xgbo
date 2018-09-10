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
