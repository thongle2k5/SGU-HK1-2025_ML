import numpy as np
from collections import Counter
class KNearestNeighbors:
    #hàm khởi tạo dữ liệu
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    #hàm lấy và lưu dữ liệu huấn luyện
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    #hàm tính toán khoản cách của 2 điểm
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
   
    #hàm dự đoán nhãn cho tập dữ liệu test
    def predict(self, X_test):
        y_pred = [self._predict_one(x) for x in X_test]
        return np.array(y_pred)
     #hàm dự đoán nhãn cho một điểm dữ liệu
    def _predict_one(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

