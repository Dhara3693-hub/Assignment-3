# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")

# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns

sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("titanic/train.csv")

plt.figure(figsize=(10, 5))
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival Count by Passenger Class (Pclass)')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.title('Survival Count by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

# %% Age distribution ?

plt.figure(figsize=(12, 6))

# KDE plot by survival
sns.kdeplot(data=train, x='Age', hue='Survived', fill=True, common_norm=False, alpha=0.5, palette=['red', 'blue'])
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Density')
plt.xlim(0, 100)  # Adjust as needed
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# %% Survived w.r.t Age distribution ?

plt.figure(figsize=(12, 6))
sns.boxplot(x='Survived', y='Age', data=train, palette='Set2')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.xticks([0, 1], ['No', 'Yes'])  # Renaming x-ticks
plt.show()


# %% SibSp / Parch distribution ?
plt.figure(figsize=(12, 6))
sns.histplot(train['SibSp'], bins=8, kde=False, color='skyblue')
plt.title('Distribution of Siblings/Spouses Aboard (SibSp)')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Frequency')
plt.xticks(range(0, 9))  # Adjusting x-ticks for clarity
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(train['Parch'], bins=8, kde=False, color='salmon')
plt.title('Distribution of Parents/Children Aboard (Parch)')
plt.xlabel('Number of Parents/Children')
plt.ylabel('Frequency')
plt.xticks(range(0, 7))  # Adjusting x-ticks for clarity
plt.show()

# %% Survived w.r.t SibSp / Parch  ?

plt.figure(figsize=(12, 6))
sns.countplot(x='SibSp', hue='Survived', data=train, palette='Set2')
plt.title('Survival Count by Number of Siblings/Spouses Aboard (SibSp)')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.xticks(range(0, train['SibSp'].max() + 1))  # Adjust x-ticks based on max SibSp
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Parch', hue='Survived', data=train, palette='Set2')
plt.title('Survival Count by Number of Parents/Children Aboard (Parch)')
plt.xlabel('Number of Parents/Children')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.xticks(range(0, train['Parch'].max() + 1))  # Adjust x-ticks based on max Parch
plt.show()

# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


log_clf = LogisticRegression()

selected_columns = ["Pclass"]
train_x = train[selected_columns]
train_y = train["Survived"]

log_clf.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(log_clf, train_x, train_y))

truth = pd.read_csv("truth_titanic.csv")
test_x = test[selected_columns]
test_y = truth["Survived"]

print(evaluate(log_clf, test_x, test_y))
# %%
