# Zachary Baum (zachary.baum.19@ucl.ac.uk),
# Wellcome EPSRC Center for Interventional and Surgical Sciences, University College London, 2020
# This code for research purposes only.
#
# Ref: 
# [1] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.2502&rep=rep1&type=pdf
#
# D_Variational [1] is defined as by Eq(20):
# D_Variational(f||g) = SUM a Wa log ((SUM a' Na' e^(-D(fa||fa')) / (SUM b Nb e^(-D(fa||gb)))
# It is considered to be most accurate of the simple, closed-form approximations provided in [1].
# To ensure the predictions are constrained by their own distribution, as well as the target, we
# take the average of this value computed in each direction.
#
# This point-pair distances in for each distribution are computed in the same way as in chamfer.py.
# See there for more information.

from keras import backend as K
import tensorflow as tf

def variational_distance(A, B, L, sigma=1.0):
	N = tf.cast(K.int_shape(L)[0], dtype=tf.float32)

	# Part 1A - D(a||a')
	a = tf.reduce_sum(tf.square(B), axis=1)
	a = tf.reshape(a, [-1, 1])
	#D_top = tf.sqrt(a - 2 * tf.matmul(B, tf.transpose(B)) + tf.transpose(a))
	D_top = a - 2 * tf.matmul(B, tf.transpose(B)) + tf.transpose(a)
	# Part 1B - SUM( e^(-D(a||a') / N )
	D_top = tf.truediv(tf.square(D_top), (2 * sigma**2))
	D_top = tf.clip_by_value(tf.exp(-D_top), 1e-15, 1e15)
	D_top = tf.truediv(D_top, N)

	# Part 2A - D(a||b)
	a = tf.reduce_sum(tf.square(B), axis=1)
	a = tf.reshape(a, [-1, 1])
	b = tf.reduce_sum(tf.square(A), axis=1)
	b = tf.reshape(b, [1, -1])
	#D_bottom = tf.sqrt(a - 2 * tf.matmul(B, tf.transpose(A)) + b)
	D_bottom = a - 2 * tf.matmul(B, tf.transpose(A)) + b
	# Part 2B - SUM( e^(-D(a||b) / N )
	D_bottom = tf.truediv(tf.square(D_bottom), (2 * sigma**2))
	D_bottom = tf.clip_by_value(tf.exp(-D_bottom), 1e-15, 1e15)
	D_bottom = tf.truediv(D_bottom, N)
	
	# Part 3 - SUM( log( 2A / 2B ) ) / N
	main_div = tf.log(tf.reduce_sum(D_top, axis=1)) - tf.log(tf.reduce_sum(D_bottom, axis=1))
	main_div = tf.reduce_sum(main_div) / N
	return K.abs(main_div)

def variational_loss_2way(y_true, y_pred):
	batched_losses_AB = tf.map_fn(lambda x: variational_distance(x[0], x[1], x[2]), (y_true, y_pred, y_pred), dtype=tf.float32)
	batched_losses_BA = tf.map_fn(lambda x: variational_distance(x[0], x[1], x[2]), (y_pred, y_true, y_pred), dtype=tf.float32)
	return K.mean(tf.stack([tf.stack(batched_losses_AB),
							tf.stack(batched_losses_BA)]))