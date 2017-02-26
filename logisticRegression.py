import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold


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
        # print test
        train_predictors = (titanic[predictors].iloc[train, :])
        train_target = titanic["Survived"].iloc[train]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(titanic[predictors].iloc[test, :])
        # print titanic[predictors].iloc[test, :]
        predictions.append(test_predictions)
    return predictions, alg


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
    submission.to_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\LogisticRegression1.csv", index_label=False,
              index=False)
    return

def main():
    titanic = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    titanic = cleaning_up_data(titanic)
    titanic = split_title(titanic)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title"]
    alg = logistic_regression(titanic, predictors)
    predictions, testData = predict_on_test_data(alg, predictors)
    generate_submission_file(predictions, testData)
    return

if __name__ == "__main__":
    main()