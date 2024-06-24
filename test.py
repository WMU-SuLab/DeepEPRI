# coding=utf-8
from model import get_model, get_model_C_sub, get_model_C_mul, get_model_max,get_model_lstm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

names = ['HeLa', 'HepG2', 'GM12878', 'IMR90', 'K562', 'hNPC', 'H1']
name=names[0]
model=None
model = get_model_lstm()
model.load_weights("./model/%sModel66.h5" % name)
save_dir = '/pub/data/pengh/DeepTest/pred/HeLa/'
predictions = {}

y_pred_list = []  # �洢Ԥ��ֵ���б�
y_test_list = []  # �洢��ʵ��ǩ���б�

#for name in names:
   
Data_dir = '/pub/data/pengh/DeepTest/data/%s/' % name
test = np.load(Data_dir+'%s_test.npz' % name)
X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']

print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
y_pred = model.predict([X_en_tes, X_pr_tes])
auc = roc_auc_score(y_tes, y_pred)
aupr = average_precision_score(y_tes, y_pred)
f1 = f1_score(y_tes, np.round(y_pred.reshape(-1)))
print("AUC : ", auc)
print("AUPR : ", aupr)
print("f1_score", f1)
         # �洢Ԥ����
        # �洢Ԥ����
predictions[name] = {'y_test': y_tes, 'y_pred': y_pred}

        # ��Ԥ��ֵ����ʵ��ǩ��ӵ��б���
y_pred_list.extend(y_pred)
y_test_list.extend(y_tes)

# ��Ԥ��ֵ����ʵ��ǩ����Ϊ�ļ�
np.save(save_dir + 'predictions_y_pred_lstm.npy', np.array(y_pred_list))
np.save(save_dir + 'predictions_y_test_lstm.npy', np.array(y_test_list))
print("save_dir")

# ��ӡԤ��������ʵ��ǩ
#for name in names:

y_test = predictions[name]['y_test']
y_pred = predictions[name]['y_pred']
print("Predictions: ", y_pred)
print("True Labels: ", y_test)
       
# ����Ԥ��������ʵ��ǩ
y_pred_array = np.load('/pub/data/pengh/DeepTest/pred/HeLa/predictions_y_pred_lstm.npy')
y_test_array = np.load('/pub/data/pengh/DeepTest/pred/HeLa/predictions_y_test_lstm.npy')

# ��Ԥ�����ת��Ϊ������Ԥ������0��1��
y_pred_binary = np.round(y_pred_array)
np.save(save_dir + 'y_pred_binary.npy', np.array(y_pred_binary))
# ����׼ȷ��
accuracy = np.mean(y_pred_binary == y_test_array)

print("Accuracy: ", accuracy)