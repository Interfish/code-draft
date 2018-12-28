import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)

c = tf.add(a, b)

with tf.Session() as sess:
    print(sess.graph.as_graph_def())