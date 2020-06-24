import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

os.chdir(r"C:\Users\Áron\Desktop\Courses\csgo-ai-competition-master\dataset_initial")


def read_dataset(template, start_idx, end_idx):
    frames = [ pd.read_json(f) for f in [template.format(i) for i in range(start_idx, end_idx+1)] ]
    return pd.concat(frames, ignore_index = True)

dftrain = read_dataset("dataset_00.json", 0, 13)
y_train = dftrain.pop('round_winner').map(lambda s: 0 if s == "CT" else 1)

dfeval = read_dataset("dataset_00.json", 13, 17)
y_eval = dfeval.pop('round_winner').map(lambda s: 0 if s == "CT" else 1)






#2
CATEGORICAL_COLUMNS = ['round_status', 'map']
NUMERIC_COLUMNS = ['round_status_time_left']
INTEGER_COLUMNS = ['alive_players_t', 'alive_players_ct', "health_t", "health_ct", "money_ct", "money_t"]

def get_attr(ds, team, attr=None):
    team_players = map(lambda players: filter(lambda p: p["team"] == team, players), ds['alive_players'])
    if attr:
        team_players = map(lambda players: map(lambda p: p[attr], players), team_players)
    
    return list(map(lambda p: list(p), team_players))

for ds in [dftrain, dfeval]:
    ds['alive_players_t']  = list(map(len ,get_attr(ds, "Terrorist")))
    ds['alive_players_ct'] = list(map(len, get_attr(ds, "CT")))
    ds['health_ct']        = list(map(sum, get_attr(ds, "CT", "health")))
    ds['health_t']         = list(map(sum, get_attr(ds, "Terrorist", "health")))
    ds['money_ct']         = list(map(sum, get_attr(ds, "CT", "money")))
    ds['money_t']          = list(map(sum, get_attr(ds, "Terrorist", "money")))

print(dftrain.info())

feature_columns = []
feature_names = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + INTEGER_COLUMNS
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))

for feature_name in INTEGER_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.int32))






#3
def make_input_fn(data_df, label_df, num_epochs=100, shuffle=True, batch_size=16):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df.filter(items=feature_names)), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
        
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
clear_output()

print(result)

tf.keras.backend.set_floatx('float64')
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')


#LAST
tf.get_logger().setLevel('WARN')

def predict_idx(data_df, idx):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices(dict(data_df.filter(items=feature_names)))
        ds = ds.skip(idx).take(1).batch(1)
        return ds
    return input_function

def predict_index(ds, idx):
    for feature in feature_names:
        print(feature, ds[feature][idx])
        
    print("")
    
    for pred in linear_est.predict(predict_idx(ds, idx)):
        probs = pred['probabilities']
        print("Win probability: CT: {:.2} T: {:.2}".format(probs[0], probs[1]))

    print("")

wt_names = linear_est.get_variable_names()
for name in wt_names:
    if name.endswith("weights"):
        val = linear_est.get_variable_value(name)
        print(name, val)

print(feature_columns)

for i in range(100,120):
    predict_index(dfeval, i)