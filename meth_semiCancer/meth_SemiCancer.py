# -*- coding: utf8 -*- 
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import os
import sys
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

# SET ENV
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
config.gpu_options.allow_growth=True

# FUNCTIONS
def fc_bn(_x, _output, _phase, _scope):
    with tf.variable_scope(_scope):
        h1 = tf.contrib.layers.fully_connected(_x, _output, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.variance_scaling_initializer(), weights_regularizer = tf.contrib.layers.l2_regularizer(0.1), reuse=tf.AUTO_REUSE)
        h2 = tf.contrib.layers.batch_norm(h1, updates_collections=None, fused=True, decay=0.9, center=True, scale=True, is_training=_phase, scope='bn', reuse=tf.AUTO_REUSE)
        return h2

x_train = pd.read_csv(sys.argv[1], dtype=np.float32).values
x_test = pd.read_csv(sys.argv[2], dtype=np.float32).values
y_train = pd.read_csv(sys.argv[3], dtype=np.float32).values
y_test = pd.read_csv(sys.argv[4], dtype=np.float32).values

x_ul = pd.read_csv(sys.argv[5], dtype=np.float32).values
y_ul = pd.read_csv(sys.argv[6], dtype=np.float32).values

unlabel_probability = float(sys.argv[7])
 
learn_rate_sm = 1e-5
learn_rate_sm_ul = 1e-3
keep_rate_sm = 0.3
keep_rate_pl = 0.3

repeated_model_num = 1
train_sm_eps = 1500
train_pl_eps = 3000

n_features = len(x_train[0])
n_classes = len(y_train[0])

# PLACEHOLDER
tf_raw_X = tf.placeholder(tf.float32, [None, n_features])
tf_raw_Y = tf.placeholder(tf.float32, [None, n_classes])
tf_raw_ul_X = tf.placeholder(tf.float32, [None, n_features])
tf_raw_ul_Y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')
handle = tf.placeholder(tf.string, shape=[])
handle2 = tf.placeholder(tf.string, shape=[])
alpha = tf.placeholder(tf.float32,)

# PARAMETERS
batch_size = 200 
prefetch_size = batch_size * 2
test_data_size = len(x_test)
ul_data_size = len(x_ul)

max_accr=0.0
sm_max_accr = 0.0

# DATASET & ITERATOR
dataset_train = tf.data.Dataset.from_tensor_slices((tf_raw_X, tf_raw_Y))
dataset_train = dataset_train.shuffle(buffer_size=batch_size * 2)
dataset_train = dataset_train.batch(batch_size)
iterator_train = dataset_train.make_initializable_iterator()

dataset_test = tf.data.Dataset.from_tensor_slices((tf_raw_X, tf_raw_Y))
dataset_test = dataset_test.batch(test_data_size)
iterator_test = dataset_test.make_initializable_iterator()
iter = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
tf_X, tf_Y = iter.get_next()

dataset_ul = tf.data.Dataset.from_tensor_slices((tf_raw_ul_X, tf_raw_ul_Y))
dataset_ul = dataset_ul.batch(ul_data_size)
iterator_ul = dataset_ul.make_initializable_iterator()

iter_ul = tf.data.Iterator.from_string_handle(handle2, dataset_ul.output_types, dataset_ul.output_shapes)
tf_ul_X, tf_ul_Y = iter_ul.get_next()

# MODEL STRUCTURE
n_in = n_features
n_sm_h1 = 1000 
n_sm_h2 = 500
n_sm_out = n_classes

print("# feature:", n_features, "# classes:", n_classes, "# train sample:", len(x_train), "# test sample:", len(x_test))

# MODEL FUNCTIONS
def softmax(_X,_keep_prob, _phase):
    fc1 = tf.nn.dropout(tf.nn.elu(fc_bn(_X, n_sm_h1, _phase, "fc1")), _keep_prob)
    fc2 = tf.nn.dropout(tf.nn.elu(fc_bn(fc1, n_sm_h2, _phase, "fc2")), _keep_prob)
    sm = tf.nn.softmax(fc_bn(fc2, n_sm_out, _phase, "sm"))
    # return sm
    return fc2, sm

# MODEL
latent, sm_out = softmax(tf_X, keep_prob, phase)
latent_ul, sm_ul_out = softmax(tf_ul_X, keep_prob, phase)

print ("MODEL READY") 

# DEFINE LOSS AND OPTIMIZER
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops) :
    sm_cost = tf.reduce_mean(-tf.reduce_sum(tf_Y * tf.log(sm_out + 1e-10), axis = 1))
    sm_ul_cost = tf.reduce_mean(-tf.reduce_sum(tf_ul_Y * tf.log(sm_ul_out + 1e-10), axis = 1))
    final_cost = tf.add(sm_cost, alpha * sm_ul_cost)
    train_op_sm = tf.train.AdamOptimizer(learning_rate=learn_rate_sm).minimize(sm_cost)
    train_op_sm_ul = tf.train.AdamOptimizer(learning_rate=learn_rate_sm_ul).minimize(final_cost)

# ACCURACY
pred = tf.argmax(sm_out, 1)
pred_pl = tf.argmax(sm_ul_out, 1)

label = tf.argmax(tf_Y, 1)
correct_pred = tf.equal(pred, label)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print ("FUNCTIONS READY")
# START SESSION
sess = tf.Session(config=config)
handle_train = sess.run(iterator_train.string_handle())
handle_test = sess.run(iterator_test.string_handle())
handle_ul = sess.run(iterator_ul.string_handle())
saver = tf.train.Saver()

a = 0.0
cur_a = 0.0
af = 0.05
t1 = 100
t2 = 200

print ("START OPTIMIZATION & TESTING")
for model_num in xrange(repeated_model_num):
    sess.run(tf.global_variables_initializer())	

    # SET OPS & FEED_DICT
    sm_ops = [sm_cost, train_op_sm, accuracy]
    pl_ops = [final_cost, train_op_sm_ul, accuracy, pred_pl, alpha]

    sm_feed_dict = {handle: handle_train, keep_prob: keep_rate_sm, phase: True}
    pl_feed_dict = {handle: handle_train, handle2: handle_ul, keep_prob: keep_rate_pl, phase: True, alpha: a}

    for temp_ep, meta_step, temp_ops, temp_feed_dict in zip([train_sm_eps, train_pl_eps], ["Pre-training", "Fine-tuning"], [sm_ops, pl_ops], [sm_feed_dict, pl_feed_dict]):	

        for ep in range(temp_ep):    
            if meta_step == "Pre-training":
                sess.run(iterator_train.initializer, feed_dict={tf_raw_X: x_train, tf_raw_Y: y_train})

                while True: 
                    try:
                        cur_cost_val, _, cur_accuracy = sess.run(temp_ops, feed_dict = pl_feed_dict)
                    except tf.errors.OutOfRangeError:
                        break

            if meta_step == "Fine-tuning" :
                pl_feed_dict = {handle: handle_train, handle2: handle_ul, keep_prob: keep_rate_pl, phase: True, alpha: a}
                sess.run(iterator_train.initializer, feed_dict={tf_raw_X: x_train, tf_raw_Y: y_train})
                if ep < t1 :
                    a = 0.0
                elif (ep >= t1) and (ep < t2) :
                    a = ((ep - t1)/(t2-t1))*af
                else :
                    a = af
                while True:
                    try: 
                        sess.run(iterator_ul.initializer, feed_dict={tf_raw_ul_X: x_ul_filter, tf_raw_ul_Y: pred_unlabel_df})
                        cur_cost_val, _, cur_accuracy, cur_pred_pl, cur_a  = sess.run(temp_ops, feed_dict = pl_feed_dict)
                    except tf.errors.OutOfRangeError:
                        break
                    
                # FILTER UNLABELED DATA USING CONFIDENCE THRESHOLD
                sess.run(iterator_ul.initializer, feed_dict={tf_raw_ul_X: x_ul, tf_raw_ul_Y: y_ul})
                cur_pred_pl, cur_ul_out = sess.run([pred_pl, sm_ul_out], feed_dict = {handle2: handle_ul, keep_prob: keep_rate_pl, phase: True})
                
                tmp = {}
                for i in range(n_classes) :
                    tmp[i] = []
                for i in range(len(cur_pred_pl)) :
                    for j in range(n_classes) :
                        if cur_pred_pl[i] == j :
                            tmp[j].append(1)
                        else :
                            tmp[j].append(0)
                pred_unlabel_df_tmp = pd.DataFrame(tmp).values	
                
                filteredpred = [[x_ul_sub, y_ul_sub, p] for x_ul_sub, y_ul_sub, p in zip(x_ul, cur_ul_out, pred_unlabel_df_tmp) if max(y_ul_sub) > unlabel_probability]
                x_ul_filter = [row[0] for row in filteredpred]
                y_ul_filter = [row[1] for row in filteredpred]
                pred_unlabel_df = [row[2] for row in filteredpred]
                
            if ep % 10 == 0:
                print("Ep:%04d," % (ep), "Cost_" + meta_step + ":%.9f" % cur_cost_val, end='')

                sess.run(iterator_test.initializer, feed_dict={tf_raw_X: x_test, tf_raw_Y: y_test})
                cur_acc, cur_pred, cur_label, cur_sm, cur_latent = sess.run([accuracy, pred, label, sm_out, latent], feed_dict = {handle: handle_test, keep_prob: 1.0, phase: False})

                print(", Train_batch_accr:%.6f" % cur_accuracy, "MAX_accr:%.6f" % max_accr, end='')

                # STORE MAX MODEL
                if max_accr < float(cur_acc):
                    max_accr = cur_acc
                    max_pred = cur_pred
                    max_label = cur_label
                    max_sm = cur_sm
                print("")

                
        print(meta_step + " part is DONE!")

        if (meta_step == "Pre-training") :
            sess.run(iterator_ul.initializer, feed_dict={tf_raw_ul_X: x_ul, tf_raw_ul_Y: y_ul})
            pred_acc, pred_unlabel = sess.run([accuracy, pred], feed_dict = {handle: handle_ul, keep_prob: 1.0, phase: False})
            tmp = {}
            for i in range(n_classes) :
                tmp[i] = []
            for i in range(len(pred_unlabel)) :
                for j in range(n_classes) :
                    if pred_unlabel[i] == j :
                        tmp[j].append(1)
                    else :
                        tmp[j].append(0)
            pred_unlabel_df = pd.DataFrame(tmp).values	
            x_ul_filter = x_ul
            sm_max_accr = max_accr
            sm_max_pred = max_pred
            max_accr = 0.0
            

# Evaluation
print("Pre-training MAX ACCURACY: " + str(sm_max_accr) + ", Fine-tuning MAX ACCURACY : " + str(max_accr))
print('Pre-training F1 SCORE : ', f1_score(max_label, sm_max_pred, average = 'weighted'), end=' ')
print('Fine-tuning F1 SCORE : ', f1_score(max_label, max_pred, average = 'weighted'))
print('Pre-training MCC SCORE : ', matthews_corrcoef(max_label, sm_max_pred), end=' ')
print('Fine-tuning MCC SCORE : ', matthews_corrcoef(max_label, max_pred))

#Save a result for test data
result = pd.DataFrame(zip(max_label, max_pred), columns=['Label', 'Predicted label'])
result.to_csv('./result_for_test_dataset.csv', index=False)
