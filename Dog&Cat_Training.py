import time
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Conv2D
#from kerastuner import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters

NAME = f'cat-vs-dog-prediction-{int(time.time())}'

tensorboard = TensorBoard(log_dir = f'logs\\NAME\\')

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X/255

print(X)
'''
def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(100,100,3)
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(2, activation='sigmoid')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

tuner_search=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output',project_name="catdogprediction")


tuner_search.search(X,y,epochs=3,validation_split=0.1)

model=tuner_search.get_best_models(num_models=1)[0]

model.summary()

model.fit(X, y, epochs=10, validation_split=0.1, initial_epoch=3)

model.save('catvsdogprediction.h5')'''