import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#x_data=np.linspace(-1,1,3000,dtype=np.float32)[:,np.newaxis]
#noise=np.random.normal(0,0.1,x_data.shape).astype(np.float32)
#y_data=np.square(x_data)-0.5+noise
"""def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
	layer_name='layer%s'%n_layer

	#with tf.name_scope('layer'):
	#	with tf.name_scope('Weights'):
	Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='w')
	tf.summary.histogram(layer_name+'/Weights',Weights)
			#tf.histogram)_summary(layer_name+'weights',Weightrs)
	#	with tf.name_scope('biases'):
	biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='bias')
	tf.summary.histogram(layer_name+'/biases',biases)
	#	with tf.name_scope('Wx_plus_b'):
	keep_prob=tf.placeholder(tf.float32)
	Wx_plus_b=tf.add(tf.matmul(inputs,Weights),biases,name='Wx_plus_b')
	Wx_plus_b=tf.nn.dropout(Wx_plus,keep_prob)	#with tf.name_score('activation'):
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)
	tf.summary.histogram(layer_name+'/outputs',outputs)
	return outputs"""
def add_layer(inputs , 
              in_size, 
              out_size,n_layer, 
              activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
         with tf.name_scope('weights'):
              Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')

              # tf.histogram_summary(layer_name+'/weights',Weights)
              tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12

         with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
              # tf.histogram_summary(layer_name+'/biase',biases)
              tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12

         with tf.name_scope('Wx_plus_b'):
              Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)

         if activation_function is None:
            outputs=Wx_plus_b
         else:
            outputs= activation_function(Wx_plus_b)

         # tf.histogram_summary(layer_name+'/outputs',outputs)
         tf.summary.histogram(layer_name + '/outputs', outputs) # Tensorflow >= 0.12

    return outputs

#with tf.name_scope('input'):
keep_prob=tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64],name='x_input')
ys=tf.placeholder(tf.float32,[None,10],name='y_input')

l1=add_layer(xs,64,30,'l1',tf.nn.relu)

prediction=add_layer(l1,30,10,'l2',tf.nn.softmax)
with tf.name_scope('loss'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)),reduction_indices=[1])) 
	#cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
	tf.summary.scalar('loss',cross_entropy)
#with tf.name_scope('train'):
train_step=tf.train.AdamOptimizer(0.3).minimize(cross_entropy)
sess=tf.Session()
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter("logs/train",sess.graph)
test_writer=tf.summary.FileWriter("logs/test",sess.graph)
sess.run(tf.global_variables_initializer())
#ax.scatter(x_data,y_dat
#plt.show()
for i in range(9000):
	sess.run(train_step,feed_dict={xs:X_train,ys:y_train})
	if i%50==0:
		train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train})
		test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test})
		train_writer.add_summary(train_result,i)
		
		test_writer.add_summary(test_result,i)
	if i%100==0:
		print i
#	if i%100==0:
#		print "finish%d"%i
#plt.ioff()
#plt.show()	`
