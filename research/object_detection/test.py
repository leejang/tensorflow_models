import numpy as np
import tensorflow as tf


indices = tf.constant([1, 3, 4])
#indices = [1, 2, 3, 5]

"""
class_encodings = [item for item in indices]
class_encodings = np.vstack(class_encodings).astype(np.int32)
print class_encodings
"""


# k-hot encoding
num_of_classes = 7
#ground_truth = tf.zeros(num_of_classes, tf.int32)
#ground_truth[indices] = 1

#ground_truth[label_index] = 1.0

res = tf.one_hot(indices=indices, depth=num_of_classes)
k_hot = tf.reduce_sum(res, 0)
#add = tf.add_n(res)

config = tf.ConfigProto(device_count = {'GPU': 0}) 
with tf.Session(config=config) as sess:
    print sess.run(res)
    print sess.run(k_hot)
    #print sess.run(ground_truth)
