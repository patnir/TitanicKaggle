from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import pandas


def get_data():
    train = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    test = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    return train, test


def random_forest(titanic, predictors, test):
    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
    print(scores.mean())
    #res = cross_validation.cross_val_predict(alg, )
    alg.fit(titanic[predictors], titanic["Survived"])
    predictions = alg.predict(test[predictors])
    return predictions


def clean_data(titanic):
    # Age
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    # Fare
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    # Sex
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    # Embarked
    titanic["Embarked"] = titanic["Embarked"].fillna(0)
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    # Getting titles
    names = titanic["Name"]
    titles = [name.split(",")[1].split(".")[0].strip() for name in names]
    titanic["Title"] = titles
    counts = titanic["Title"].value_counts().index.values
    for i in range(len(counts)):
        titanic.loc[titanic["Title"] == counts[i], "Title"] = i
    titanic["Title"] = titanic["Title"].fillna(0)
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
    return titanic


def generate_submission_file(predictions, data):
    submission = pandas.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived": predictions
    })
    print(submission)
    submission.to_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\RandomForest1.csv", index_label=False,
              index=False)
    return

def main():
    train, test = get_data()
    train = clean_data(train)
    test = clean_data(test)
    predictors = ["Pclass", "Sex", "Age", "Embarked", "SibSp", "Title"]
    predictions = random_forest(train, predictors, test)

    generate_submission_file(predictions, test)

    return

if __name__ == '__main__':
    main()