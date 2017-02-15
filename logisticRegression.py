import pandas
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

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
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=5)
    print(scores.mean())
    return alg


def test_on_data(alg, predictors):
    titanicTest = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    titanicTest = cleaning_up_data(titanicTest)
    titanicTest = split_title(titanicTest)
    return

def main():
    titanic = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    titanic = cleaning_up_data(titanic)
    titanic = split_title(titanic)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title"]
    alg = logistic_regression(titanic, predictors)
    test_on_data(alg, predictors)
    return

if __name__ == "__main__":
    main()