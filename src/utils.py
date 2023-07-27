import pandas as pd

def preprocess_iris():
    df = pd.read_csv("Iris.csv")
    X_iris = df.drop(["Species", "Id"], axis=1)
    return X_iris