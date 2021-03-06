import pandas
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import glob
from sklearn import cross_validation
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

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
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
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


def find_next_file(filename):
    files = [f for f in glob.glob(filename + "*")]
    n = len(files) + 1
    return filename + str(n) + ".csv"


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
    filen = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\LinearRegression"
    dirn = find_next_file(filen)
    df.to_csv(dirn, index_label=False,
              index=False)
    return

def identify_best_predictors(titanic, predictors):

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(titanic[predictors], titanic["Survived"])

    # Get the raw p-values for each feature, and transform them from p-values into scores
    scores = -np.log10(selector.pvalues_)

    # Plot the scores
    # Do you see how "Pclass", "Sex", "Title", and "Fare" are the best features?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

    # Pick only the four best features
    predictors = ["Pclass", "Sex", "Fare", "Title"]

    alg = LinearRegression()
    # Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())

def family_group(titanic):
        family_id_mapping = {}

        def get_family_id(row):
            last_name = row["Name"].split(",")[0]
            family_id = "{0}{1}".format(last_name, row["FamilySize"])
            if family_id not in family_id_mapping:
                if len(family_id_mapping) == 0:
                    current_id = 1
                else:
                    current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
                family_id_mapping[family_id] = current_id
            return family_id_mapping[family_id]

        family_ids = titanic.apply(get_family_id, axis=1)
        family_ids[titanic["FamilySize"] < 3] = -1
        # print(pandas.value_counts(family_ids))
        titanic["FamilyId"] = family_ids
        return titanic


def main():
    titanic = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    titanic = CleaningUpCode(titanic)

    titanic = splitTitle(titanic)

    #predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title"]
    #predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "NameLength"]

    predictors = ["Pclass", "Sex", "Embarked", "FamilySize", "Title", "NameLength", "Fare"]

    titanic = family_group(titanic)

    #predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]

    identify_best_predictors(titanic, predictors)

    linearPredictionsOnTrain, alg = linearRegression(titanic, predictors)

    maxThreshold, maxAccuracy = getBestThresholdValue(titanic, linearPredictionsOnTrain)

    print maxThreshold, maxAccuracy

    testOnDataSet(alg, maxThreshold, predictors)

    return

if __name__ == "__main__":
    main()