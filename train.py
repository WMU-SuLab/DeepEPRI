import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
from datetime import datetime
import tensorflow as tf
import model
from model import DeepEPRI, set_seed
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import summary as hparams_summary


class roc_callback(Callback):
    def __init__(self, val_data, name, csv_file):
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.name = name
        self.csv_file = csv_file
        self.csv_writer = csv.writer(csv_file)
        self.csv_writer.writerow(['epoch', 'auc_val', 'aupr_val','acc_val', 'precision_val','recall_val','f1_val','learning_rate'])

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en, self.pr])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        f1_val = f1_score(self.y, np.round(y_pred.reshape(-1)))
        learning_rate = self.model.optimizer.lr.numpy()
        acc_val = logs.get('accuracy')  # 获取准确率
        precision_val = logs.get('precision')  # 获取精确率
        recall_val = logs.get('recall')  # 获取召回率
        self.model.save_weights("./model/%s_Weights%d.h5" % (self.name, epoch), overwrite=True)

        print('\r auc_val: %s ' % str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        print('\r f1_val: %s ' % str(round(f1_val, 4)), end=100 * ' ' + '\n')

        # Write validation metrics to CSV file #
        self.csv_writer.writerow([epoch, auc_val ,aupr_val,acc_val, precision_val, recall_val, f1_val,learning_rate])
        self.csv_file.flush()
        # Write validation metrics to tensorboard #

        #summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tf.summary.create_file_writer(self.log_dir).as_default():
            # with summary_writer.as_default():
            tf.summary.scalar('auc_val', auc_val, step=epoch)
            tf.summary.scalar('aupr_val', aupr_val, step=epoch)
            tf.summary.scalar('f1_val', f1_val, step=epoch)
       # summary_writer.flush()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

model.set_seed(2024)
names = ['All'，'HeLa', 'HepG2', 'GM12878', 'IMR90', 'K562', 'hNPC', 'H1']
name = names[1]
# The data used here is the sequence processed by data_processing.py.
Data_dir = './data/%s/' % name
train = np.load(Data_dir + '%s_train.npz' % name)
# test=np.load(Data_dir+'%s_test.npz'%name)
X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']

X_en_tra, X_en_val, X_pr_tra, X_pr_val, y_tra, y_val = train_test_split(
    X_en_tra, X_pr_tra, y_tra, test_size=0.1, stratify=y_tra, random_state=250)

log_dir = "./logs/HeLa/20250319/%s" % name  # 日志目录的路径
tensorboard = TensorBoard(log_dir=log_dir)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
csv_file_path = './metrics_DeepEPRI.csv'

model = None
model = get_DeepEPRI()
# 修改权重的句柄名称
for i in range(len(model.weights)):
    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)

model.summary()
print('Traing %s cell line specific model ...' % name)

with open(csv_file_path, mode='w', newline='') as csv_file:
    back = roc_callback(val_data=[X_en_val, X_pr_val, y_val], name=name,csv_file=csv_file)
    back.log_dir = log_dir
    history = model.fit([X_en_tra, X_pr_tra], y_tra, validation_data=([X_en_val, X_pr_val], y_val), epochs=200,
                        batch_size=32, callbacks=[back, tensorboard, early_stopping_monitor,reduce_lr])
# Close the CSV file
    csv_file.close()
    #stopped_epoch = early_stopping_monitor.stopped_epoch
t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("开始时间:" + t1 + "结束时间：" + t2)
#print("早停的轮数:", history.stopped_epoch)




