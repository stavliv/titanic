from sklearn import svm
from data import get_train_val_test
from sklearn.metrics import accuracy_score

train_data, val_data, test_data = get_train_val_test("train.csv", "test.csv", "gender_submission.csv")

clf = svm.SVC(kernel='poly', degree=3)
clf.fit(train_data[0], train_data[1])

pr = clf.predict(val_data[0])
print(f'Val set accuracy: {accuracy_score(val_data[1], pr)}')

prt = clf.predict(test_data[0])
print(f' set accuracy: {accuracy_score(test_data[1], prt)}')

