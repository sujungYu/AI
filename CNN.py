import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./mnist/data/",one_hot=True)

X=tf.placeholder(tf.float32,[None, 28, 28, 1])
Y=tf.placeholder(tf.float32, [None, 10])
keep_prob=tf.placeholder(tf.float32)

global_step=tf.Variable(0,trainable=False,name='global_step')

with tf.name_scope('layer1'):
  W1=tf.Variable(tf.random_normal([5,5,1,32], stddev=0.01))
  L1=tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='SAME')
  L1=tf.nn.relu(L1)
  L1=tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('layer2'):
  W2=tf.Variable(tf.random_normal([5,5,32,64], stddev=0.01))
  L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
  L2=tf.nn.relu(L2)
  L2=tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('layer2'):
  W3=tf.Variable(tf.random_normal([7*7*64, 256], stddev=0.01))
  L3=tf.reshape(L2,[-1,7*7*64])
  L3=tf.matmul(L3,W3)
  L3=tf.nn.relu(L3)
  L3=tf.nn.dropout(L3, keep_prob)

with tf.name_scope('output'):
  W4=tf.Variable(tf.random_normal([256,10],stddev=0.01))
  model=tf.matmul(L3,W4)
  tf.summary.histogram("Weights", W4)
  tf.summary.histogram("Model", model)

with tf.name_scope('optimizer'):
  cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
  optimizer=tf.train.AdamOptimizer(0.001).minimize(cost,global_step=global_step)
  tf.summary.scalar('cost',cost)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./logs',sess.graph)

batch_size=100
total_batch=int(mnist.train.num_examples/batch_size)

for epoch in range(100):
  total_cost=0
 
  for i in range(total_batch):
    batch_xs, batch_ys=mnist.train.next_batch(batch_size)
    batch_xs=batch_xs.reshape(-1,28,28,1)
   
    _,cost_val=sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
   
    total_cost+=cost_val
    summary=sess.run(merged,feed_dict={X: batch_xs, Y : batch_ys, keep_prob: 0.7})
    writer.add_summary(summary,epoch)
   
  print('Epoch:', '%04d'%(epoch + 1), 'Avg. cost=','{:.3f}'.format(total_cost/total_batch))
 
print('finish')

is_correct=tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy:',  sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1,28,28,1), Y:mnist.test.labels, keep_prob:1}))
