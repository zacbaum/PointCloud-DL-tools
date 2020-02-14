# Zachary Baum (zachary.baum.19@ucl.ac.uk),
# Wellcome EPSRC Center for Interventional and Surgical Sciences, University College London, 2020
# This code for research purposes only.
#
# Ref: 
# [1] http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf
# [2] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.2502&rep=rep1&type=pdf
# [3] https://en.wikipedia.org/wiki/Euclidean_distance
#
# Chamfer Distance [1] is defined as:
# d_CD(S1, S2) = SUM (for x in S1) of the minimum (for y in S2) ||x - y||^2 + 
#				 SUM (for y in S2) of the minimum (for x in S1) ||x - y||^2
# It is useful for measuring the discrepancy between two unordered sets of points.
# For distributions, this is near to D_min Eq(10) [2].

from keras import backend as K
import tensorflow as tf

def chamfer_distance(S1, S2):
	# Compute the point pair distances using the equivalency from Eq(2) [3].
	row_norms_S1 = tf.reduce_sum(tf.square(S1), axis=1)
	row_norms_S1 = tf.reshape(row_norms_true, [-1, 1])
	row_norms_S2 = tf.reduce_sum(tf.square(S2), axis=1)
	row_norms_S2 = tf.reshape(row_norms_pred, [1, -1])
	D = row_norms_S1 - 2 * tf.matmul(S1, S2, False, True) + row_norms_S2
	# Obtain chamfer distance as in above equation.
	dist_x_to_y = K.mean(K.min(D, axis=0))
	dist_y_to_x = K.mean(K.min(D, axis=1))
	dist = K.mean(tf.stack([dist_y_to_x, dist_x_to_y]))
	return dist

def chamfer_loss(y_true, y_pred):
	batched_losses = tf.map_fn(lambda x: chamfer_distance(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
	return K.mean(tf.stack(batched_losses))