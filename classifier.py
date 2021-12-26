from sklearn.svm import LinearSVC
import joblib 
import glob
import os
import time


train_path = './data/features/train/'
test_path = './data/features/test/'

if __name__ == "__main__":
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_path, '*.feat')):
        data = joblib.load(feat_path)
        # 分别拿数据和标签
        fds.append(data[:-1])
        labels.append(data[-1])
    if clf_type == 'LIN_SVM':
        # 定义分类器
        clf = LinearSVC(max_iter=1000)
        print("Training a Linear SVM Classifier......")
        # 训练
        clf.fit(fds, labels)
        print('Training done!')
        print('Testing......')
        # 测试，计算精度
        for feat_path in glob.glob(os.path.join(test_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            temp = data_test[:-1]
            data_test_feat = temp.reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num) / total
        t1 = time.time()
        print('classification accuracy is :%f' % rate)
        print('total time is :%f' % (t1 - t0))
