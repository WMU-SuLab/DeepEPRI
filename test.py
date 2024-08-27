# coding=utf-8
from model import get_model, get_model_lstm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    accuracy_score
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

weights = ['mix', 'HeLa', 'HepG2', 'GM12878', 'IMR90', 'K562', 'hNPC', 'H1']
weight = weights[0]
model = None
# model = get_model()
model = get_model_lstm()
model.load_weights("./model/k7/mixModel_LSTMbcel56.weights.h5")  #
print("already load %s_model" % weight)
save_dir = './pred/mix_lstm_bcel_k7/'

threshold = 0.5
predictions = {}

# 定义存储指标的字典
metrics_dict = {}

# 定义各个细胞系数据的行范围
cell_lines = {
    'HeLa': [(1, 4384), (13304, 21793)], 1w3左右
    'H1': [(4385, 4893), (21794, 22842)],1500左右
    'hNPC': [(4894, 8315), (22843, 29795)],1w左右
    'GM12878': [(8316, 11001), (29796, 35360)],
    'K562': [(11002, 12093), (35361, 37522)],
    'HepG2': [(12094, 12607), (37523, 38564)],
    'IMR90': [(12608, 13303), (38565, 39910)],

}

for cell_line, ranges in cell_lines.items():
    y_pred_list = []
    y_test_list = []
    y_pred_binary_list = []
    for idx, data_range in enumerate(ranges):
        start_row, end_row = data_range
        X_en_tes = X_pr_tes = y_tes = None


        Data_dir = './data/%s/k7/' % weight
        test = np.load(Data_dir + '%s_test.npz' % weight)
        X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']


        X_en_tes = X_en_tes[start_row - 1:end_row]
        X_pr_tes = X_pr_tes[start_row - 1:end_row]
        y_tes = y_tes[start_row - 1:end_row]

        if X_en_tes is not None and X_pr_tes is not None and y_tes is not None:
            print("****************Testing %s cell line specific model on rows %d to %d****************" % (
            cell_line, start_row, end_row))
            try:
                y_pred = model.predict([X_en_tes, X_pr_tes])
                y_pred_binary = np.where(y_pred >= threshold, 1, 0)

            # 计算指标
                auc = roc_auc_score(y_tes, y_pred)
                aupr = average_precision_score(y_tes, y_pred)
                accuracy = accuracy_score(y_tes, y_pred_binary)
                precision = precision_score(y_tes, y_pred_binary)
                recall = recall_score(y_tes, y_pred_binary)
                f1 = f1_score(y_tes, y_pred_binary)

            # 打印指标时加上细胞系名称
                print("AUC for %s: %f" % (cell_line, auc))
                print("ACC for %s: %f" % (cell_line, accuracy))
                print("AUPR for %s: %f" % (cell_line, aupr))
                print("Precision for %s: %f" % (cell_line, precision))
                print("Recall for %s: %f" % (cell_line, recall))
                print("F1 Score for %s: %f" % (cell_line, f1))

            # 存储指标到字典中
                if cell_line not in metrics_dict:
                    metrics_dict[cell_line] = {'AUC': [], 'ACC': [], 'AUPR': [], 'Precision': [], 'Recall': [], 'F1': []}
                    metrics_dict[cell_line]['AUC'].append(auc)
                    metrics_dict[cell_line]['ACC'].append(accuracy)
                    metrics_dict[cell_line]['AUPR'].append(aupr)
                    metrics_dict[cell_line]['Precision'].append(precision)
                    metrics_dict[cell_line]['Recall'].append(recall)
                    metrics_dict[cell_line]['F1'].append(f1)

            # 将预测值和真实标签添加到列表中
                y_pred_list.extend(y_pred)
                y_test_list.extend(y_tes)
                y_pred_binary_list.extend(y_pred_binary)
            except ValueError as e:
                print("An error occurred during prediction for %s cell line: %s" % (cell_line, str(e)))

    # 将预测值和真实标签保存为文件
    np.save(save_dir + 'predictions_y_pred_%s.npy'% cell_line, np.array(y_pred_list))
    np.save(save_dir + 'predictions_y_test_%s.npy' % cell_line, np.array(y_test_list))
    np.save(save_dir + 'predictions_y_pred_binary_%s.npy' % cell_line, np.array(y_pred_binary_list))
    print("save_dir is : ", save_dir)

    # 打印所有细胞系的指标
for cell_line, metrics in metrics_dict.items():
    print("Metrics for %s:" % cell_line)
    print("Average AUC: %f" % np.mean(metrics['AUC']))
    print("Average ACC: %f" % np.mean(metrics['ACC']))
    print("Average AUPR: %f" % np.mean(metrics['AUPR']))
    print("Average Precision: %f" % np.mean(metrics['Precision']))
    print("Average Recall: %f" % np.mean(metrics['Recall']))
    print("Average F1 Score: %f" % np.mean(metrics['F1']))
    # 打印预测结果和真实标签
    #y_test = predictions[cell_line]['y_test']
    #y_pred = predictions[cell_line]['y_pred']
    #y_pred_binary = predictions[cell_line]['y_pred_binary']
    #print("Predictions_%s: ", y_pred)
    #print("True Labels_%s: " , y_test)
    #print("Predictions_binary_%s: " , y_pred_binary)

