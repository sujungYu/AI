import tensorflow as tf
import tensorflow as random_uniform

x_data=[1,2,3,4,5]
y_data=[1.5,3,5.5,9,13.5]

a=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.random_uniform([1],-1.0,1.0))

X= tf.placeholder(tf.float32,name="X")
Y= tf.placeholder(tf.float32,name="Y")

hypothesis = a*X*X+b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.002)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
   
    for step in range(5000):
      _, cost_val = sess.run([train_op,cost], feed_dict={X:x_data,Y:y_data})
      print(step,cost_val,sess.run(a),sess.run(b))
   
    print("X:5,Y:",sess.run(hypothesis,feed_dict={X:5}))
    print("X:2.5,Y:",sess.run(hypothesis,feed_dict={X:2.5}))
   
