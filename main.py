from src.dbscan import DBScan
from src.kmeans import KMeans
from src.kmedoids import KMedoids
from src.utils import *

print("Masukkan algoritma yang ingin digunakan:")
print("\t1. kmeans")
print("\t2. kmedoids")
print("\t3. DBScan")
while True:
    algo = int(input())
    if algo < 1 or algo > 3:
        print("Masukan tidak valid")
    else:
        break
    
print("Masukkan dataset yang ingin digunakan:")
filename = input()

X = preprocess_iris()
model = None

if algo == 3:
    while True:
        print("Masukkan nilai epsilon (eps > 0):")
        eps = float(input())
        if eps <= 0:
            print("Masukan tidak sesuai (eps > 0)")
        else:
            break
    
    while True:
        print("Masukkan minimal points (min_samples >= 1):")
        min_samples = int(input())
        model = DBScan(eps=eps, min_samples=min_samples)
        if min_samples < 1:
            print("Masukan tidak sesuai (min_samples >= 1)")
        else:
            break
    
else:
    while True:
        print("Masukkan nilai k (k >= 1):")
        k = int(input())
        if k < 1:
            print("Masukan tidak sesuai (k >= 1)")
        else:
            break
    
    if algo == 1:
        model = KMeans(n_clusters=k)
    elif algo == 2:
        model = KMedoids(n_clusters=k)

model.fit(X)
print("\n================================= RESULT =================================")
print(model.labels)
X['label'] = model.labels
print(X)