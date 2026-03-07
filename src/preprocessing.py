import pandas as pd

def load_and_split_data(file_path):
    data = pd.read_csv(file_path)

    train_data = data.iloc[:200000]
    new_data = data.iloc[200000:]

    X_train = train_data.drop("Class", axis=1)
    y_train = train_data["Class"]

    X_new = new_data.drop("Class", axis=1)
    y_new = new_data["Class"]

    return X_train, y_train, X_new, y_new