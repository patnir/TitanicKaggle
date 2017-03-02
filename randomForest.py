from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import operator
import matplotlib.pyplot as plt
import glob
import pandas
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def get_data():
    train = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    test = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    return train, test


def random_forest(titanic, predictors, test):
    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
    print(scores.mean())
    alg.fit(titanic[predictors], titanic["Survived"])
    res = alg.predict(test[predictors])
    return res


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


def generate_submission_file(predictions, data):
    submission = pandas.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived": predictions
    })
    print(submission)

    filen = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\RandomForest"
    dirn = find_next_file(filen)

    submission.to_csv(dirn, index_label=False, index=False)
    return


def find_next_file(filename):
    files = [f for f in glob.glob(filename + "*")]
    n = len(files) + 1
    return filename + str(n) + ".csv"


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

    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
    # Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())


def main():
    train, test = get_data()
    train = clean_data(train)
    test = clean_data(test)
    train = family_group(train)
    test = family_group(test)

    # predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title",
    # "FamilyId", "NameLength"]

    predictors = ["Pclass", "Sex", "Embarked", "FamilySize", "Title", "NameLength", "Fare"]

    predictions = random_forest(train, predictors, test)

    identify_best_predictors(train, predictors)

    generate_submission_file(predictions, test)

    return

if __name__ == '__main__':
    main()