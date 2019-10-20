import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('acc')>0.998):
			print("99.8 percent accuracy reached")
			self.model.stop_training = True


mycallback = myCallback()

mnist = tf.keras.datasets.mnist
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

train_img = train_img.reshape(60000,28,28,1)
test_img = test_img.reshape(10000,28,28,1)

train_img = train_img/255.0
test_img = test_img / 255.0

model = tf.keras.Sequential([
							tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),
							tf.keras.layers.MaxPooling2D(2,2),
							tf.keras.layers.Flatten(),
							tf.keras.layers.Dense(128, activation='relu'),
							tf.keras.layers.Dense(10, activation='softmax')

	])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_img, train_labels, epochs= 100, callbacks = [mycallback])
