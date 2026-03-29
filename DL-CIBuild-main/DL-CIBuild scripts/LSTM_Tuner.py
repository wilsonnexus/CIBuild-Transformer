try:
    from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, rand
    HYPEROPT_AVAILABLE = True
except ImportError:
    hp = Trials = STATUS_OK = fmin = tpe = rand = None
    HYPEROPT_AVAILABLE = False

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import optunity
import optunity.metrics
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import Utils
import GA.GARunner as GARunner
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from timeit import default_timer as timer


def train_preprocess(dataset_train, time_step):
    training_set = dataset_train.iloc[:, 0:19].values
    if Utils.with_smote:
        X = training_set
        y = dataset_train.iloc[:, 0].values
        X, y = SMOTE().fit_resample(X, y)
        training_set = X

    X_train = []
    y_train = []
    for i in range(time_step, len(training_set)):
        X_train.append(training_set[i-time_step:i, 0])
        y_train.append(training_set[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train


def test_preprocess(dataset_train, dataset_test, time_step):
    y_test = dataset_test.iloc[:, 0:1].values
    dataset_total = pd.concat((dataset_train['build_Failed'], dataset_test['build_Failed']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
    inputs = inputs.reshape(-1, 1)

    X_test = []
    for j in range(time_step, len(inputs)):
        X_test.append(inputs[j-time_step:j, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test, y_test


def get_threshold_list(dataset):
    cdt = dataset['build_Failed'] > 0
    failure_rate = (dataset[cdt].shape[0] / dataset.shape[0])
    return list(Utils.frange(0.01, max(1, failure_rate), 0.1))


class LSTMWorker(Worker):
    def __init__(self, train_set, **kwargs):
        super().__init__(**kwargs)
        self.train_set = train_set

    def compute(self, config, *args, **kwargs):
        res = construct_lstm_model(config, self.train_set)
        return {
            'loss': float(res["validation_loss"]),
            'info': res["entry"]
        }


def construct_lstm_model(network_params, train_set):
    X_train, y_train = train_preprocess(train_set, network_params["time_step"])
    drop = round(network_params["drop_proba"], 2)

    classifier = Sequential()
    classifier.add(LSTM(units=network_params["nb_units"], return_sequences=True, input_shape=(X_train.shape[1], 1)))
    classifier.add(Dropout(drop))

    for _ in range(0, network_params["nb_layers"]):
        classifier.add(LSTM(units=network_params["nb_units"], return_sequences=True))
        classifier.add(Dropout(drop))

    classifier.add(LSTM(units=network_params["nb_units"]))
    classifier.add(Dropout(drop))
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(
        optimizer=network_params["optimizer"],
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )

    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=10)

    result = classifier.fit(
        X_train,
        y_train,
        epochs=network_params["nb_epochs"],
        batch_size=network_params["nb_batch"],
        verbose=0,
        callbacks=[es]
    )

    validation_loss = np.amin(result.history['loss'])
    entry = Utils.predict_lstm(classifier, X_train, y_train)
    entry['validation_loss'] = validation_loss

    return {
        'validation_loss': validation_loss,
        'model': classifier,
        'entry': entry
    }


global data
global global_params
global global_model
global global_entry


def train_lstm_with_hyperopt(network_params):
    global global_params, global_model, global_entry
    res = construct_lstm_model(network_params, data)
    global_params = network_params
    global_model = res["model"]
    global_entry = res["entry"]
    return {
        'loss': res['validation_loss'],
        'status': STATUS_OK,
    }


def convert_from_PSO(network_params):
    for key in network_params:
        if key == 'optimizer':
            if int(network_params[key]) == 1:
                network_params[key] = 'adam'
            else:
                network_params[key] = 'rmsprop'
        elif 'drop_proba' not in key and 'decision_threshold' not in key:
            network_params[key] = int(network_params[key])
    return network_params


def fn_lstm_pso(drop_proba=0.01, nb_units=32, nb_epochs=2, nb_batch=4, nb_layers=1, optimizer=1, time_step=30):
    if int(optimizer) == 1:
        optimizer = 'adam'
    else:
        optimizer = 'rmsprop'

    network_params = {
        'nb_units': int(nb_units),
        'nb_layers': int(nb_layers),
        'optimizer': optimizer,
        'time_step': int(time_step),
        'nb_epochs': int(nb_epochs),
        'nb_batch': int(nb_batch),
        'drop_proba': drop_proba
    }
    res = construct_lstm_model(network_params, data)
    return 1 - float(res["validation_loss"])


def evaluate_tuner(tuner_option, train_set):
    global data
    data = train_set

    nb_units = [32, 64]
    nb_epochs = [4, 5, 6]
    nb_batch = [4, 8, 16, 32, 64]
    nb_layers = [1, 2, 3, 4]
    optimizers = ['adam', 'rmsprop']
    time_steps = list(range(30, 61))
    drops = [round(x, 2) for x in np.arange(0.01, 0.21, 0.01)]

    space_tpe = None
    if HYPEROPT_AVAILABLE:
        space_tpe = {
            'drop_proba': hp.choice('drop_proba', drops),
            'nb_units': hp.choice('nb_units', nb_units),
            'nb_epochs': hp.choice('nb_epochs', nb_epochs),
            'nb_batch': hp.choice('nb_batch', nb_batch),
            'nb_layers': hp.choice('nb_layers', nb_layers),
            'optimizer': hp.choice('optimizer', optimizers),
            'time_step': hp.choice('time_step', time_steps)
        }

    start = timer()

    if "tpe" in tuner_option:
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is not installed, so tuner_option='tpe' cannot be used.")
        trials = Trials()
        fmin(train_lstm_with_hyperopt, space_tpe, algo=tpe.suggest, max_evals=Utils.max_eval, trials=trials)
        best_params = global_params
        best_model = global_model
        entry_train = global_entry

    elif "ga" in tuner_option:
        rnn_param_choices = {
            'nb_units': nb_units,
            'nb_layers': nb_layers,
            'optimizer': optimizers,
            'time_step': time_steps,
            'nb_epochs': nb_epochs,
            'nb_batch': nb_batch,
            'drop_proba': drops
        }
        best_params, best_model, entry_train = GARunner.generate(rnn_param_choices, construct_lstm_model, data)

    elif "rs" in tuner_option:
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is not installed, so tuner_option='rs' cannot be used.")
        trials = Trials()
        fmin(train_lstm_with_hyperopt, space_tpe, algo=rand.suggest, max_evals=Utils.max_eval, trials=trials)
        best_params = global_params
        best_model = global_model
        entry_train = global_entry

    else:
        best_params = {
            'nb_units': 64,
            'nb_layers': 3,
            'optimizer': 'adam',
            'time_step': 30,
            'nb_epochs': 10,
            'nb_batch': 64,
            'drop_proba': 0.1
        }
        res = construct_lstm_model(best_params, data)
        entry_train = res["entry"]
        best_model = res["model"]

    end = timer()
    entry_train["time"] = end - start
    entry_train["params"] = best_params
    entry_train["model"] = best_model
    return entry_train