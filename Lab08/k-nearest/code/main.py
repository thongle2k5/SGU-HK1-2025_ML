import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from K_Nearest_Neighbors import KNearestNeighbors 
import time

def bt1():
    df = pd.read_csv('dataset/Iris.csv')
    print("--- 5 dòng đầu tiên của dữ liệu ---")
    print(df.head())
    if 'Id' in df.columns:
       df = df.drop('Id', axis=1)
    X = df.iloc[:, :-1].values 
    y = df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    k_value = 3
    clf = KNearestNeighbors(k=k_value)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"\n=> Độ chính xác nhận diện loài hoa (Accuracy): {accuracy * 100:.2f}%")
    return


# Bài tập 2------------------------------------------------------------------------------------------
def bt2():
   data=pd.read_csv('dataset/letter-recognition.csv')
   print(data.head())
   X = data.iloc[:, 1:].values 
   y = data.iloc[:,0].values
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   k_value = 5
   clf = KNearestNeighbors(k=k_value)
   clf.fit(X_train, y_train)
   predictions = clf.predict(X_test[:1000]) # giảm kích thước để tính nhanh hơn
   accuracy = np.sum(predictions == y_test[:1000]) / 1000
   print(f"\nĐộ chính xác của nhận diện chữ : {accuracy * 100:.2f}%")
   return
start_time = time.time()
bt2()
end_time = time.time()
print(f"\nThời gian thực thi: {end_time - start_time} giây")
