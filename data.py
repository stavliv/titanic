import pandas as pd
import numpy as np
from numpy import nanmean
from sklearn.model_selection import train_test_split

def prepare_data(csv):
    data = pd.read_csv(csv)
    data.Sex = data.Sex.apply(lambda x: int(x=="male"))
    simple = data.drop(["Ticket", "Cabin"], axis=1)
    simple.Embarked = simple.Embarked.replace({"S" : 0, "C" : 1, "Q" : 2})

    means = np.zeros(3)
    for counter, value in enumerate(means):
        pc = simple[simple.Pclass == (counter + 1)]
        means[counter] = nanmean(pc.Fare)

    for counter, value in enumerate(means):
        simple.Fare.mask(simple.Pclass == counter + 1 & simple.Fare.isna(), value, inplace=True)

    simple.Age.mask(("Master" in simple.Name) & simple.Age.isna(), 5, inplace=True)
    simple.Age.mask(("Miss" in simple.Name) & (simple.Parch != 0) & simple.Age.isna(), 5, inplace=True)
    simple.Age.interpolate(inplace=True)
    simple.drop(["PassengerId"], axis=1, inplace=True)
    simple.drop(["Name"], axis=1, inplace=True)
    simple.interpolate(inplace=True)
    return simple

def prepare_train_data(train_csv):
    df = prepare_data(train_csv)
    y_train = df.Survived.values
    y_train = y_train.astype(np.float32)
    y_train = [[value] for value in y_train]
    x = df.drop(["Survived"], axis=1)
    normalized_x=(x-x.mean())/x.std()
    x_train = normalized_x.values
    x_train = x_train.astype(np.float32)
    return [x_train, y_train]

def prepare_test_data(test_csv, test_y_csv):
    df = prepare_data(test_csv)
    normalized_df=(df-df.mean())/df.std()
    x_test = normalized_df.values
    x_test = x_test.astype(np.float32)
    y = pd.read_csv(test_y_csv)
    y_test = y.Survived.values
    y_test = y_test.astype(np.float32)
    y_test = [[value] for value in y_test]
    return [x_test, y_test]

def get_train_val_test(train_csv, test_csv, test_y_csv):
    train_data = prepare_train_data(train_csv)
    x_train, x_val, y_train, y_val = train_test_split(*train_data, test_size=0.2)
    train_data = [x_train, y_train]
    val_data = [x_val, y_val]
    test_data = prepare_test_data(test_csv, test_y_csv)
    return [train_data, val_data, test_data]

def get_train_test(train_csv, test_csv, test_y_csv):
    train_data = prepare_train_data(train_csv)
    test_data = prepare_test_data(test_csv, test_y_csv)
    return [train_data, test_data]
