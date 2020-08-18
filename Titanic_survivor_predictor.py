import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# This is from a Kaggle competition where one tries to find out the survivors from Titanic disaster.
# I strongly recommend participating this competition yourself! My record is 0.79 accuracy.
# https://www.kaggle.com/c/titanic


# Read the data given from the platform and check how the data looks like.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
gender_reveal = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(train_data.head())

# The data has to be enchanced a bit... for example add missing data (Age)

common_titles = ['Mr.','Miss.','Mrs.','Master.']  # Titles found from dataset for grouping.

# Go through each dataset, since there's data missing in both.
for df in [train_data, test_data]:
    
    df['FamilySize'] = df['Parch']+df['SibSp']+1  # Create a new column for how many familymembers person had.
    
    # Set binary for travellers without any company :( .
    df['IsAlone']=0 
    df.loc[(df.FamilySize==1),'IsAlone'] = 1
    
    # Rich people tend to survive, create a column for name length.
    df['NameLen'] = df['Name'].apply(lambda x : len(x))
    
    # Take the possible title from name (first part of split) and set it into own column. If not public, set to Misc.
    df['Title'] = df['Name'].apply(lambda x : x.split(',')[1].strip().split()[0])
    df['Title'] = df['Title'].apply( lambda x : x if x in common_titles else 'Misc.')
    
    # Regarding fare and embarked, just copy from mean and mode. There's no better way to group this...
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    

# Fill missing ages by taking a mean of said title.
titlewise_grouping = train_data.groupby('Title')
train_data['Age'] = titlewise_grouping['Age'].apply(lambda x: x.fillna(x.mean()))

# For test data as well.
titlewise_grouping = test_data.groupby('Title')
test_data['Age'] = titlewise_grouping['Age'].apply(lambda x: x.fillna(x.mean()))

# Represent sex, title and class using numbers (easier for the model to work).
for dataframe in [train_data,test_data]:
    
    dataframe['Sex'] = dataframe['Sex'].map( {'male':0, 'female': 1 } )
    dataframe['Title'] = dataframe['Title'].map( {'Mr.':1, 'Misc.': 0, 'Master.':2, 'Miss.': 3, 'Mrs.': 4 } )
    dataframe['Pclass'] = dataframe['Pclass'].map( {1:3,2:2,3:1} )


# Now to the actual model training when data is preprocessed...

# Get just the labels of the samples for model.
y = train_data["Survived"]

# Which features to use in this version.
features = ["Pclass","Sex", "Fare", "FamilySize", "IsAlone", "NameLen", "Title", "Embarked"]

# Form pandas dummies.
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)

y_pred=clf.predict(X_test)

# Since we are dealing with binary (survived/didn't survive), we have to round up predictions to correct format.
predictions = y_pred > 0.5
predictions = predictions*1
predictions = np.array(predictions)
predictions = predictions.flatten()

# Deal with output.
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_new.csv', index=False)
print("Your submission was successfully saved!")
