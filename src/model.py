import tensorflow as tf
from keras import layers, models


def build_model(img_width=512, img_height=128, num_chars=80):
	# Input layer
	input_img = layers.Input(shape=(img_height, img_width, 1), name="image")

	# Convolutional layers
	x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
	x = layers.MaxPooling2D((2, 2))(x)

	# Reshape for LSTM layers
	x = layers.Reshape((-1, x.shape[-1]))(x)

	# Bidirectional LSTM layers
	x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

	# Dense layer with softmax activation for CTC
	x = layers.Dense(num_chars)(x)
	x = layers.Activation('softmax')(x)

	# Define model
	model = models.Model(inputs=input_img, outputs=x)

	def ctc_loss(y_true, y_pred):
		# CTC expects the time-major format (time_steps, batch_size, num_classes)
		y_pred = tf.transpose(y_pred, perm=[1, 0, 2])

		# Compute input and label lengths
		input_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_pred)[0])
		label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), dtype=tf.int32), axis=1)

		# Debugging: print the shapes of tensors
		tf.print("y_true shape:", tf.shape(y_true))
		tf.print("y_pred shape:", tf.shape(y_pred))
		tf.print("input_length shape:", tf.shape(input_length))
		tf.print("label_length shape:", tf.shape(label_length))

		# Calculate CTC loss
		loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
		return tf.reduce_mean(loss)

	model.compile(optimizer="adam", loss=ctc_loss, metrics=[])  # CTC doesn't use accuracy metrics
	return model
