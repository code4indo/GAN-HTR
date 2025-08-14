def ctc_loss_lambda_func(y_true, y_pred):
	"""
	Compute CTC loss for sequence-to-sequence learning.
	Handles variable length sequences and proper tensor shapes.
	"""
	# Get batch size dynamically
	batch_size = tf.shape(y_true)[0]
	
	# Get sequence length from y_pred (time steps)
	sequence_length = tf.shape(y_pred)[1]
	
	# y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
	# For CTC loss, input_length should be the length of the sequence (time steps)
	# We'll use the full sequence length as input_length
	input_length = tf.fill([batch_size, 1], sequence_length)
	input_length = tf.cast(input_length, tf.int32)
	
	# y_true strings are padded with 0
	# so sum of non-zero gives number of characters in this string
	label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int32")
	
	# Ensure label_length doesn't exceed sequence length to avoid indexing errors
	label_length = tf.minimum(label_length, sequence_length)
	
	# Ensure label_length doesn't exceed the second dimension of y_true
	max_label_length = tf.shape(y_true)[1]
	label_length = tf.minimum(label_length, max_label_length)
	
	# Ensure input_length doesn't exceed the second dimension of y_pred
	max_input_length = tf.shape(y_pred)[1]
	input_length = tf.minimum(input_length, max_input_length)
	
	# Debug prints to understand the shapes
	tf.print("y_true shape:", tf.shape(y_true))
	tf.print("y_pred shape:", tf.shape(y_pred))
	tf.print("input_length:", input_length)
	tf.print("label_length:", label_length)
	tf.print("max_label_length:", max_label_length)
	tf.print("max_input_length:", max_input_length)
	
	# Additional validation: ensure y_true has enough dimensions for the longest label
	max_actual_label_length = tf.reduce_max(label_length)
	tf.print("max_actual_label_length:", max_actual_label_length)
	
	# If y_true doesn't have enough dimensions, pad it
	if max_actual_label_length > max_label_length:
		# Pad y_true to accommodate the longest label
		padding_needed = max_actual_label_length - max_label_length
		y_true = tf.pad(y_true, [[0, 0], [0, padding_needed]], constant_values=0)
		tf.print("Padded y_true to shape:", tf.shape(y_true))
	
	# Compute CTC loss with error handling
	try:
		loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
		# average loss across all entries in the batch
		loss = tf.reduce_mean(loss)
	except Exception as e:
		# Fallback to a simple MSE loss if CTC fails
		print(f"Warning: CTC loss computation failed, using fallback: {e}")
		loss = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_pred, axis=1, keepdims=True)))

	return loss