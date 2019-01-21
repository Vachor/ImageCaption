import tensorflow as tf

sess = tf.InteractiveSession()

weight = tf.get_variable(name='weight',shape=[4096,100],initializer=tf.glorot_uniform_initializer())
sess.run(tf.initialize_all_variables())

tf.reset_default_graph()

sess = tf.InteractiveSession()
bias = tf.get_variable(name='weight',shape=[4096,100],initializer=tf.glorot_uniform_initializer())
sess.run(tf.initialize_all_variables())