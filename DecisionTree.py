from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Data import get_data
from sklearn.metrics import accuracy_score

def decision_tree(train_data, test_data):
    dt = DecisionTreeClassifier(criterion = 'entropy')
    dt.fit(train_data[0], train_data[1])
    dt_pred_train = dt.predict(test_data[0])
    print(f'Test set accuracy: {accuracy_score(test_data[1], dt_pred_train)}')

def random_forest(train_data, test_data):
    rfc = RandomForestClassifier(criterion = 'entropy')
    rfc.fit(train_data[0], train_data[1])
    rfc_pred_train = rfc.predict(test_data[0])
    print(f'Test set accuracy: {accuracy_score(test_data[1], rfc_pred_train)}')

if __name__ == '__main__':
    train_data, val_data, test_data = get_data("train.csv", "test.csv", "gender_submission.csv")
    #decision_tree(train_data, test_data)
    random_forest(train_data, test_data)
