import tensorflow as tf
import keras
import numpy as np

def house_model(y_new):
    xs = np.array([0, 1, 2, 4, 6, 8, 10], dtype=float) # Your Code Here#
    ys = np.array([0.50, 0.100, 1.50, 2.50, 3.50, 4.50, 5.50], dtype=float) # Your Code Here#
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # Your Code Here#
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs,ys, epochs=100)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)

