from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np
import pandas
import glob


def get_data():
    train = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\mtrain.csv")
    test = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\mtest.csv")
    return train, test


def gradient_boosing(titanic, titanic_test):
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
         ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
    ]

    kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

    predictions = []
    for train, test in kf:
        train_target = titanic["Survived"].iloc[train]
        full_test_predictions = []
        # Make predictions for each algorithm on each fold
        for alg, predictors in algorithms:
            # Fit the algorithm on the training data
            alg.fit(titanic[predictors].iloc[train, :], train_target)
            # Select and predict on the test fold
            # We need to use .astype(float) to convert the dataframe to all floats and avoid an sklearn error
            test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
            full_test_predictions.append(test_predictions)
        # Use a simple ensembling scheme&#8212;just average the predictions to get the final classification
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
        # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction
        test_predictions[test_predictions <= .5] = 0
        test_predictions[test_predictions > .5] = 1
        predictions.append(test_predictions)

    # Put all the predictions together into one array
    predictions = np.concatenate(predictions, axis=0)

    # Compute accuracy by comparing to the training data
    accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
    print(accuracy)

    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
    ]

    full_predictions = []
    for alg, predictors in algorithms:
        # Fit the algorithm using the full training data.
        alg.fit(titanic[predictors], titanic["Survived"])
        # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error
        predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
        full_predictions.append(predictions)

    # The gradient boosting classifier generates better predictions, so we weight it higher
    predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1

    print(predictions)


def find_next_file(filename):
    files = [f for f in glob.glob(filename + "*")]
    # numbers = [int(re.findall(r"[\d]+", i)[-1]) for i in files]
    # n = max(numbers) + 1
    n = len(files) + 1
    return filename + str(n) + ".csv"


def generate_submission_file(titanic, titanic_test):
    predictors = ["Pclass", "Sex", "Embarked", "FamilySize", "Title", "NameLength", "Fare"]

    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Embarked", "FamilySize", "Title", "NameLength", "Fare"]]
    ]

    full_predictions = []
    for alg, predictors in algorithms:
        # Fit the algorithm using the full training data.
        alg.fit(titanic[predictors], titanic["Survived"])
        # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error
        predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
        full_predictions.append(predictions)

    # The gradient boosting classifier generates better predictions, so we weight it higher
    predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1
    predictions = predictions.astype(int)
    submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    print(submission.describe())
    filen = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\Submissions\\GradientBoosting"
    dirn = find_next_file(filen)
    submission.to_csv(dirn, index_label=False, index=False)

def main():
    train, test = get_data()
    #gradient_boosing(train, test)
    generate_submission_file(train, test)



if __name__ == '__main__':
    main()
