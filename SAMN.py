from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_wine
import datetime
import warnings
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")
T = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 为了实验可重复，网络参数初始化固定，需注意不同的初始化会影响实验结果（可调整）
def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset):
    if dataset == 1:
        iris = load_iris()
        Y = iris.target
        X = iris.data
        X = np.array(X)
        y = np.array(Y).flatten().astype(float)
        random_state = 42

    elif dataset == 2:
        wine = load_wine()
        Y = wine.target
        X = wine.data
        X = np.array(X)
        y = np.array(Y).flatten().astype(float)
        random_state = 42

    elif dataset == 3:
        wine_quality = pd.read_csv(filepath_or_buffer="bigdata/winequality-red.csv", header=0, sep=';')
        wine_quality.dropna(inplace=True)
        wine_quality_data = wine_quality.iloc[:, :-1]
        wine_quality_target = wine_quality['quality'] - 3
        X = np.array(wine_quality_data)
        y = np.array(wine_quality_target).flatten().astype(float)
        random_state = 42

    elif dataset == 4:
        dba_data = pd.read_csv(filepath_or_buffer="bigdata/Dry_Bean_Dataset.csv", header=0)
        dba_data.loc[dba_data['Class'] == 'SEKER', 'Class'] = 0
        dba_data.loc[dba_data['Class'] == 'BARBUNYA', 'Class'] = 1
        dba_data.loc[dba_data['Class'] == 'BOMBAY', 'Class'] = 2
        dba_data.loc[dba_data['Class'] == 'CALI', 'Class'] = 3
        dba_data.loc[dba_data['Class'] == 'DERMASON', 'Class'] = 4
        dba_data.loc[dba_data['Class'] == 'HOROZ', 'Class'] = 5
        dba_data.loc[dba_data['Class'] == 'SIRA', 'Class'] = 6
        dba_data.dropna(inplace=True)
        Y = dba_data['Class']
        X = dba_data.drop(columns=['Class'])
        X = np.array(X)
        y = np.array(Y).flatten().astype(float)
        random_state = 42

    elif dataset == 5:
        bws_data = pd.read_csv(filepath_or_buffer="bigdata/batteryless_wearable_sensor_DataSet.csv", header=0)
        bws_data['8'] -= 1
        bws_data.dropna(inplace=True)
        Y = bws_data['8']
        X = bws_data.drop(columns=['8'])
        X = np.array(X)
        y = np.array(Y).flatten().astype(float)
        random_state = 42

    elif dataset == 6:
        accel_data = pd.read_csv(filepath_or_buffer="bigdata/accelerometer.csv", header=0)
        accel_data['wconfid'] -= 1
        accel_data.dropna(inplace=True)
        Y = accel_data['wconfid']
        X = accel_data.drop(columns=['wconfid'])
        X = np.array(X)
        y = np.array(Y).flatten().astype(float)
        random_state = 42

    return X, y, random_state


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, X_trains, y_trains):
        self.len = X_trains.shape[0]
        self.x_data = torch.FloatTensor(X_trains)
        self.y_data = torch.LongTensor(y_trains)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class SAMN(nn.Module):
    def __init__(self, x_features, classnum, mid_features):
        super(SAMN, self).__init__()
        self.classnum = classnum
        self.fc1 = nn.Linear(x_features, x_features)
        self.fc2 = nn.Linear(x_features, x_features)
        self.fc3 = nn.Linear(x_features, mid_features * x_features)
        self.fc4 = nn.Linear(x_features, mid_features * x_features)
        self.fc5 = nn.Linear(x_features * mid_features, classnum)
        self.fc6 = nn.Linear(x_features, x_features)
        self.fc7 = nn.Linear(mid_features * x_features, mid_features * x_features)
        self.fc8 = nn.Linear(mid_features * x_features, mid_features * x_features)
        self.fc9 = nn.Linear(mid_features * x_features, mid_features * x_features)
        self.fc10 = nn.Linear(2 * x_features, x_features)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(x_features, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True, device=None, dtype=None)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 50))
        self.labels = []
        # 类原型初始化
        for i in range(0, self.classnum):
            self.labels.append(torch.tensor([0] * mid_features * x_features))
        self.label_0 = torch.tensor([0] * mid_features * x_features)
        self.trains = True
        self.y_trains = []
        self.batch_label = []
        self.c1 = torch.rand(x_features)
        self.c2 = torch.rand(x_features)
        self.hs = []
        # 传递信息h_i初始化
        for i in range(0, self.classnum):
            self.hs.append(torch.rand(mid_features * x_features))

    def forward(self, x):
        # 特征提取模块
        out = self.fc1(x)  # 一层
        out = self.relu(out)
        out = self.fc2(out)  # 一层
        out = self.relu(out)

        # 注意力机制输入
        out1 = self.fc3(out)
        self.label_0 = self.tanh(out1)

        # 进一步特征提取模块
        out = self.fc4(out)  # 一层
        out = self.tanh(out)

        if self.trains:
            outs = []
            for i in range(self.classnum):
                if (len(self.label_0[self.batch_label == i]) != 0):
                    self.labels[i] = self.label_0[self.batch_label == i]
                    self.outlabel_11 = torch.mm(self.labels[i], self.labels[i].transpose(1, 0))
                    self.outlabel_11 = torch.softmax(self.outlabel_11, dim=1)
                    self.labels[i] = torch.mm(self.outlabel_11, self.labels[i])
                    self.labels[i] = torch.mean(self.labels[i], dim=0)
                    self.hs[i] = self.sigmoid(self.fc7(self.labels[i]) + self.fc8(self.hs[i]))
                    self.hs[i].detach_()
                    self.labels[i] = self.tanh(self.fc9(self.hs[i]))

                # Inner-class
                s = torch.cosine_similarity(out, self.labels[i].unsqueeze(0), dim=-1)
                if (outs == []):
                    outs = s.unsqueeze(0)
                else:
                    outs = torch.cat([outs, s.unsqueeze(0)], 0)

            # Inter-class:这里是类原型MLP，可替换成consine计算方式
            ls = []
            for i in range(self.classnum):
                if (ls == []):
                    ls = self.labels[i].unsqueeze(0)
                else:
                    ls = torch.cat([ls, self.labels[i].unsqueeze(0)], 0)
            ls = self.fc5(ls)
            ls = self.tanh(ls)

            # Inner-class+Inter-class,构成总loss
            out = torch.cat([outs.transpose(1, 0), ls], 0)
            return out

        return out


if __name__ == '__main__':
    batches = [128, 128, 128, 512, 512, 512]
    cs = [3, 3, 6, 7, 4, 3]
    for num in range(1, 2):
        bs = batches[num - 1]
        init_seeds(seed=42)
        X, y, rs = load_data(dataset=num)
        print("dataset:", X.shape, y.shape)
        best_train_scores = []
        best_valid_scores = []
        best_test_scores = []
        trainacc_5 = []
        testacc_5 = []
        validacc_5 = []
        testprecison_5 = []
        testrecall_5 = []
        testf1score_5 = []

        # 多次数据集划分，取均值
        kf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=rs)
        for train_index, test_index in kf.split(X, y):
            train_scores = []
            valid_scores = []
            test_scores = []
            n_bestvalid = 0
            n_besttest = 0
            bestepoch_validacc = 0
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 验证集划分
            X_trains, X_valid, y_trains, y_valid = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42)

            scaler = StandardScaler()  # 数据预处理，使得经过处理的数据符合正态分布，即均值为0，标准差为1
            X_trains = scaler.fit_transform(X_trains)
            X_valid = scaler.transform(X_valid)
            X_test = scaler.transform(X_test)

            # 创建 trainloader 和 testloader
            trainset = TrainDS(X_trains, y_trains)
            train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=bs, shuffle=True,
                                                       num_workers=0)
            # test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=0)

            # 初始化网络
            model = SAMN(X_trains.shape[1], cs[num - 1], 4)
            model.y_trains = y_trains

            # optimizer是训练的工具，传入net的所有参数和学习率
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            # print(optimizer)
            criterion = nn.CrossEntropyLoss(reduction='none')
            norms_Xtrain = []
            norms_Xvalid = []
            totalloss = []
            totaliter = []
            bestn_10 = []
            X_trains = torch.as_tensor(torch.from_numpy(np.array(X_trains)), dtype=torch.float32).to(
                device=device)
            X_valid = torch.as_tensor(torch.from_numpy(np.array(X_valid)), dtype=torch.float32).to(
                device=device)
            X_test = torch.as_tensor(torch.from_numpy(np.array(X_test)), dtype=torch.float32).to(
                device=device)
            bestepoch_validacc = 0
            curloss = 0

            # y_trains = np.concatenate([y_trains, np.array([0, 1])])
            y_trains = torch.as_tensor(torch.from_numpy(np.array(y_trains)), dtype=torch.long).to(
                device=device)
            # torch.autograd.set_detect_anomaly = True
            # with torch.autograd.detect_anomaly():
            for i in range(1000):
                Output_train = None
                Labels = None
                for j, (inputs, labels) in enumerate(train_loader):
                    model.trains = True
                    model.train()
                    X_trains.requires_grad_()

                    model = model.to(device)
                    model.batch_label = labels
                    output_train = model(inputs)

                    # 类原型label
                    labels = torch.as_tensor(torch.from_numpy(np.array(labels)), dtype=torch.long).to(
                        device=device)
                    labels = torch.cat([labels, torch.from_numpy(np.array([k for k in range(cs[num - 1])]))], 0)

                    if Output_train == None:
                        Output_train = output_train
                        Labels = labels
                    else:
                        Output_train = torch.cat([Output_train, output_train], 0)
                        Labels = torch.cat([Labels, labels], 0)

                    output_train.requires_grad_()

                    model.train()
                    loss = criterion(output_train, labels)
                    optimizer.zero_grad()
                    sample_weight = torch.ones(labels.shape[0])
                    sample_weight[-cs[num - 1]:] = 10  # 提高类原型分类权重，weight为10
                    loss = loss * sample_weight
                    loss = loss.mean()

                    totalloss.append(loss.item())
                    loss = loss
                    loss.requires_grad_(True)
                    loss.backward(retain_graph=True)

                    # 更新权重参数
                    optimizer.step()

                y_trains = Labels.detach().cpu().numpy()
                output_train = torch.argmax(Output_train, dim=1)

                with torch.no_grad():
                    model.trains = False
                    model.eval()
                    output_valid = model(X_valid)
                    output_test = model(X_test)

                output_valids = []
                for p in range(cs[num - 1]):
                    s_valid = torch.cosine_similarity(output_valid, model.labels[p].unsqueeze(0), dim=-1)
                    if (output_valids == []):
                        output_valids = s_valid.unsqueeze(0)
                    else:

                        output_valids = torch.cat([output_valids, s_valid.unsqueeze(0)], 0)
                output_valid = torch.argmax(output_valids, dim=0)

                output_tests = []
                for q in range(cs[num - 1]):
                    s_test = torch.cosine_similarity(output_test, model.labels[q].unsqueeze(0), dim=-1)
                    if (output_tests == []):
                        output_tests = s_test.unsqueeze(0)
                    else:
                        output_tests = torch.cat([output_tests, s_test.unsqueeze(0)], 0)
                output_test = torch.argmax(output_tests, dim=0)

                train_acc = accuracy_score(y_trains, output_train.detach().cpu().numpy()) * 100
                valid_acc = accuracy_score(y_valid, output_valid.detach().cpu().numpy()) * 100
                test_acc = accuracy_score(y_test, output_test.detach().cpu().numpy()) * 100

                test_pre = precision_score(y_test, output_test.detach().cpu().numpy(), average='macro') * 100
                test_recall = recall_score(y_test, output_test.detach().cpu().numpy(), average='macro') * 100
                test_f1 = f1_score(y_test, output_test.detach().cpu().numpy(), average='macro') * 100

                train_scores.append(np.around(train_acc, 4))
                valid_scores.append(np.around(valid_acc, 4))
                test_scores.append(np.around(test_acc, 4))

                # 采用验证集挑选最佳epoch的模型，记录测试结果
                if (valid_acc >= bestepoch_validacc):
                    bestepoch_validacc = valid_acc
                    bestepoch_testacc = test_acc
                    bestepoch_trainacc = train_acc
                    bestepoch_testpre = test_pre
                    bestepoch_testrecall = test_recall
                    bestepoch_testf1 = test_f1

            print('best epoch train acc:', np.round(bestepoch_trainacc, 4))
            print('best epoch valid acc:', np.round(bestepoch_validacc, 4))
            print('best epoch test acc:', np.round(bestepoch_testacc, 4))
            print('best epoch test precision:', np.round(bestepoch_testpre, 4))
            print('best epoch test recall:', np.round(bestepoch_testrecall, 4))
            print('best epoch test f1 score:', np.round(bestepoch_testf1, 4))

            # 多次实验的记录结果
            trainacc_5.append(bestepoch_trainacc)
            testacc_5.append(bestepoch_testacc)
            validacc_5.append(bestepoch_validacc)
            testprecison_5.append(bestepoch_testpre)
            testrecall_5.append(bestepoch_testrecall)
            testf1score_5.append(bestepoch_testf1)

        print('5 folds test acc:',
              str(np.round(np.array(trainacc_5).mean(), 2)) + "\u00B1" + str(
                  np.round(np.array(trainacc_5).std(), 2)),
              str(np.round(np.array(testacc_5).mean(), 2)) + "\u00B1" + str(
                  np.round(np.array(testacc_5).std(), 2)),
              trainacc_5, validacc_5, testacc_5)
        print('5 folds test precision:',
              str(np.round(np.array(testprecison_5).mean(), 2)) + "\u00B1" + str(
                  np.round(np.array(testprecison_5).std(), 2)),
              testprecison_5)
        print('5 folds test recall:',
              str(np.round(np.array(testrecall_5).mean(), 2)) + "\u00B1" + str(
                  np.round(np.array(testrecall_5).std(), 2)),
              testrecall_5)
        print('5 folds test f1 score:',
              str(np.round(np.array(testf1score_5).mean(), 2)) + "\u00B1" + str(
                  np.round(np.array(testf1score_5).std(), 2)),
              testf1score_5)
        # 保存实验结果
        # f = open('result/' + str(T) + '-sat-' + str(num) + '-mid3-result.txt', mode='a')  # 打开文件，若文件不存在系统自动创建。
        # f.write('dateset:')  # write 写入
        # f.write(str(num) + '\n')
        # f.write('bs:' + str(bs) + '\n' + str(trainacc_5) + '\n' + str(testacc_5) + '\n')
        # f.write(
        #     str(np.round(np.array(trainacc_5).mean(), 2)) + "\u00B1" + str(
        #         np.round(np.array(trainacc_5).std(), 2)) + '  ' + str(
        #         np.round(np.array(testacc_5).mean(), 2)) + "\u00B1" + str(
        #         np.round(np.array(testacc_5).std(), 2)) + '\n')
        # f.write(
        #     str(np.round(np.array(testprecison_5).mean(), 2)) + "\u00B1" + str(
        #         np.round(np.array(testprecison_5).std(), 2)) + '\n')
        # f.write(
        #     str(np.round(np.array(testrecall_5).mean(), 2)) + "\u00B1" + str(
        #         np.round(np.array(testrecall_5).std(), 2)) + '\n')
        # f.write(
        #     str(np.round(np.array(testf1score_5).mean(), 2)) + "\u00B1" + str(
        #         np.round(np.array(testf1score_5).std(), 2)) + '\n')
        # f.close()  # 关闭文件
