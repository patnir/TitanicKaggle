import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np
import glob

def cleaning_up_data(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    titanic["Embarked"] = titanic["Embarked"].fillna(0)
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    return titanic


def split_title(titanic):
    names = titanic["Name"]
    titles = [name.split(",")[1].split(".")[0].strip() for name in names]
    # pPrint(titles)
    titanic["Title"] = titles
    # print titanic["Embarked"].unique()
    counts = titanic["Title"].value_counts().index.values
    for i in range(len(counts)):
        titanic.loc[titanic["Title"] == counts[i], "Title"] = i
    titanic["Title"] = titanic["Title"].fillna(0)
    return titanic


def logistic_regression(titanic, predictors):
    alg = LogisticRegression(random_state=1)
    alg.fit(titanic[predictors], titanic["Survived"])
    return alg

def logistic_regression_with_cv(titanic, predictors):
    alg = LogisticRegression()
    kf = KFold(titanic.shape[0], n_folds=5, random_state=1)
    predictions = []
    for train, test in kf:
        train_predictors = (titanic[predictors].iloc[train, :])
        train_target = titanic["Survived"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(titanic[predictors].iloc[test, :])
        predictions.append(test_predictions)

    maxThreshold, maxAccuracy = getBestThresholdValue(titanic, predictions)

    print(maxThreshold, maxAccuracy)

    return alg, maxThreshold


def testPredictions(titanic, predictions, threshold):
    testPredictions = np.concatenate(predictions, axis=0)
    testPredictions[testPredictions > threshold] = 1
    testPredictions[testPredictions <= threshold] = 0
    matches = sum([testPredictions == titanic["Survived"]])
    accuracy = float(sum(matches)) / float(len(matches))
    return accuracy

def getBestThresholdValue(titanic, predictions):
    threshold = 0.0
    maxThreshold = 0.0
    maxAccuracy = 0.0
    for i in range(0, 100):
        accuracy = testPredictions(titanic, predictions, threshold)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            maxThreshold = threshold
        threshold = i / 100.00
    return maxThreshold, maxAccuracy


def predict_on_test_data(alg, predictors):
    titanicTest = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    titanicTest = cleaning_up_data(titanicTest)
    titanicTest = split_title(titanicTest)
    predictions = alg.predict(titanicTest[predictors])
    return predictions, titanicTest

def generate_submission_file(predictions, data):
    submission = pandas.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived": predictions
    })
    print(submission)
    submission.to_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\LogisticRegression2CV.csv", index_label=False,
              index=False)
    return


def find_next_file(filename):
    files = [f for f in glob.glob(filename + "*")]
    # numbers = [int(re.findall(r"[\d]+", i)[-1]) for i in files]
    # n = max(numbers) + 1
    n = len(files) + 1
    return filename + str(n) + ".csv"


def testOnDataSet(alg, maxThreshold, predictors):
    titanicTest = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    titanicTest = cleaning_up_data(titanicTest)
    titanicTest = split_title(titanicTest)
    predictions = alg.predict(titanicTest[predictors])
    # print titanicTest[predictors].isnull().any()

    predictions[predictions > maxThreshold] = 1
    predictions[predictions <= maxThreshold] = 0
    # print len(predictions)
    predictions = list(predictions)
    ids = range(892, 892 + len(predictions) + 1)
    res = list(zip(ids, predictions))
    # print res
    df = pandas.DataFrame(data=res, columns=["PassengerId", "Survived"])
    df["Survived"] = df["Survived"].astype(int)
    filen = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\LogisticRegression"
    dirn = find_next_file(filen)
    df.to_csv(dirn, index_label=False,
              index=False)
    return


def main():
    titanic = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    titanic = cleaning_up_data(titanic)
    titanic = split_title(titanic)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title"]
    alg, maxThreshold = logistic_regression_with_cv(titanic, predictors)

    # predictions, testData = predict_on_test_data(alg, predictors)

    # generate_submission_file(predictions, testData)

    testOnDataSet(alg, maxThreshold, predictors)

    return

if __name__ == "__main__":
    main()