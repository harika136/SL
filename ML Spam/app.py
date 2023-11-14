from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('praveen.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_mail_data = pd.read_csv('spam_mail_data.csv')
        raw_mail_data.loc[raw_mail_data['Category'] == 'spam', 'Category',] = 0
        raw_mail_data.loc[raw_mail_data['Category'] == 'ham', 'Category',] = 1
        X = raw_mail_data['Message']
        Y = raw_mail_data['Category']
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=3)
        feature_extraction = TfidfVectorizer(
            min_df=1, stop_words='english', lowercase=True)
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')
        model = SVC()
        model.fit(X_train_features, Y_train)
        prediction_on_training_data = model.predict(X_train_features)
        accuracy_on_training_data = accuracy_score(
            Y_train, prediction_on_training_data)
        prediction_on_test_data = model.predict(X_test_features)
        accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
        # input_mail = [" I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
        input_mail = request.form['input_mail']
        input_data_features = feature_extraction.transform([input_mail])
        prediction = model.predict(input_data_features)
        if (prediction[0] == 1):
            k = 'Ham mail'
        else:
            k = 'Spam mail'

    return render_template("praveen.html", k=k)


if __name__ == "__main__":
    app.run(debug=True)
