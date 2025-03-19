# coding=utf-8
from model import DeepEPRI
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    accuracy_score
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

weights = ['All', 'HeLa', 'HepG2', 'GM12878', 'IMR90', 'K562', 'hNPC', 'H1']
weight = weights[0]
model = None
model = DeepEPRI()
model.load_weights("./model/bestmodel.h5")  #
print("already load %s_model" % weight)
save_dir = './pred/'

threshold = 0.5
predictions = {}

t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#for name in names:
#names = ['All','HeLa', 'HepG2', 'GM12878', 'IMR90', 'K562', 'hNPC', 'H1']
names = ['HepG2']
for name in names:   
    Data_dir = './data/%s/6cell_test/' % name
    test = np.load(Data_dir+'%s_test_6cell.npz' % name)
    X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']

    print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
    y_pred = model.predict([X_en_tes, X_pr_tes])
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)
    print("y_pred_binary.shape",y_pred_binary.shape)
    #y_pred_rounded = np.round(y_pred.reshape(-1))
    #print("y_pred_rounded.shape",y_pred_rounded.shape)
    print("y_tes.shape",y_tes.shape)
    auc = roc_auc_score(y_tes, y_pred)
    aupr = average_precision_score(y_tes, y_pred)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    print("AUC : ", auc)
    print("ACC : ", accuracy)
    print("AUPR : ", aupr)
    print("Precision : ",precision)
    print("Recall : ",recall)
    print("F1 Score : ", f1)
         
  
    np.save(save_dir + 'y_pred_tesindata_%s.npy' % name, y_pred)
    np.save(save_dir + 'y_tes_%s.npy' % name, y_tes)
    np.save(save_dir + 'y_pred_binary_tesindata_%s.npy' % name, y_pred_binary)
    print("save_dir is : ",save_dir)

    # 打印预测结果和真实标签
    #y_test = predictions[name]['y_test']
    #y_pred = predictions[name]['y_pred']
    #y_pred_binary = predictions[name]['y_pred_binary']
    #print("Predictions_%s: " % name, y_pred)
    #print("True Labels_%s: " % name, y_test)
    #print("Predictions_binary_%s: " % name, y_pred_binary)
       
  
t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("start_time:" + t1 + "end_time:" + t2)

