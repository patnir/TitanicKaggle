import pandas
import operator


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


def get_data():
    train = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\train.csv")
    test = pandas.read_csv("C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\test.csv")
    return train, test


def store_test_train(train, test):
    trainfile = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\mtrain.csv"
    testfile = "C:\Users\SUNITA\Desktop\HackBaby!\TitanicKaggle\\mtest.csv"
    train.to_csv(trainfile, index_label=False, index=False)
    test.to_csv(testfile, index_label=False, index=False)


def main():
    train, test = get_data()
    train = clean_data(train)
    test = clean_data(test)
    train = family_group(train)
    test = family_group(test)
    store_test_train(train, test)


if __name__ == '__main__':
    main()
