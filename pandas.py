import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')
print(df.head())
# df.info()

df.columns

y=df['diabetes']
X =df.drop(['diabetes'], axis=1)
#X=df[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi','dpf', 'age']]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

model=LinearRegression()

model.fit(X_train,y_train)

print(model.predict(X_test))

print(model)

user_input = {
    'pregnancies': 6,
    'glucose': 148,
    'diastolic': 72,
    'triceps': 35,
    'insulin': 0,
    'bmi': 33.6,
    'dpf': 0.627,
    'age': 50
}

user_df = pd.DataFrame([user_input])

predicted_diabetes = model.predict(user_df)

if predicted_diabetes[0] == 1:
    print("The model predicts that the user has diabetes.")
else:
    print("The model predicts that the user does not have diabetes.")

print("ewwwwwwwwwww")

