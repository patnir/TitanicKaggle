import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

def linearRegression(titanic, predictors):
    alg = LinearRegression()
    kf = KFold(titanic.shape[0], n_folds=5, random_state=1)
    predictions = []
    for train, test in kf:
        #print test
        train_predictors = (titanic[predictors].iloc[train, :])
        train_target = titanic["Survived"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(titanic[predictors].iloc[test, :])
        #print titanic[predictors].iloc[test, :]
        predictions.append(test_predictions)
    return predictions, alg

def testPredictions(titanic, predictions, threshold):
    testPredictions = np.concatenate(predictions, axis=0)
    testPredictions[testPredictions > threshold] = 1
    testPredictions[testPredictions <= threshold] = 0
    matches = sum([testPredictions == titanic["Survived"]])
    accuracy = float(sum(matches)) / float(len(matches))
    return accuracy

def CleaningUpCode(titanic):
    # print(titanic.describe())
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    # print titanic["Embarked"].unique()
    titanic["Embarked"] = titanic["Embarked"].fillna(0)
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    # print titanic.describe()
    return titanic

def getBestThresholdValue(titanic, linearPredictionsOnTrain):
    threshold = 0.0
    maxThreshold = 0.0
    maxAccuracy = 0.0
    for i in range(0, 100):
        accuracy = testPredictions(titanic, linearPredictionsOnTrain, threshold)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            maxThreshold = threshold
        threshold = i / 100.00
    return maxThreshold, maxAccuracy

def pPrint(data):
    for i in data:
        print i

def splitTitle(titanic):
    names = titanic["Name"]
    titles = [name.split(",")[1].split(".")[0].strip() for name in names]
    #pPrint(titles)
    titanic["Title"] = titles
    #print titanic["Embarked"].unique()
    counts = titanic["Title"].value_counts().index.values
    for i in range(len(counts)):
        titanic.loc[titanic["Title"] == counts[i], "Title"] = i
    titanic["Title"] = titanic["Title"].fillna(0)
    return titanic

def testOnDataSet(alg, maxThreshold, predictors):
    titanicTest = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    titanicTest = CleaningUpCode(titanicTest)
    titanicTest = splitTitle(titanicTest)
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
    df.to_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\LinearRegression1.csv", index_label=False,
              index=False)
    return

def main():
    titanic = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    titanic = CleaningUpCode(titanic)

    titanic = splitTitle(titanic)

    #predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title"]
    predictors = ["Pclass", "Sex", "Age", "Embarked", "SibSp"]
    linearPredictionsOnTrain, alg = linearRegression(titanic, predictors)

    maxThreshold, maxAccuracy = getBestThresholdValue(titanic, linearPredictionsOnTrain)

    print maxThreshold, maxAccuracy

    testOnDataSet(alg, maxThreshold, predictors)

    return

if __name__ == "__main__":
    main()