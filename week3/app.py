import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectPercentile, chi2
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\first inten project.csv")
df.columns = df.columns.str.strip()


df.drop(columns=['Booking_ID', 'date of reservation'], inplace=True)


numerical_atts = ["number of adults", "number of children", "number of weekend nights", 
                  "number of week nights", "lead time", "P-C", "P-not-C", "special requests", "average price"]

def handle_outliers_with_mean(df, numerical_atts):
    Q1 = df[numerical_atts].quantile(0.25)
    Q3 = df[numerical_atts].quantile(0.75)
    IQR = Q3 - Q1

    for col in numerical_atts:
        mean = df[col].mean()
        df.loc[df[col] < (Q1[col] - 1.5 * IQR[col]), col] = mean
        df.loc[df[col] > (Q3[col] + 1.5 * IQR[col]), col] = mean

    return df


df = handle_outliers_with_mean(df, numerical_atts)


df[numerical_atts] = df[numerical_atts].astype(int)


label_encoder = LabelEncoder()
df['repeated'] = label_encoder.fit_transform(df['repeated'])


df['car parking space'] = label_encoder.fit_transform(df['car parking space'])


df = pd.get_dummies(df, columns=['room type', 'type of meal', 'market segment type'])


ordinal_mapping = {'Not_Canceled': 0, 'Canceled': 1}
df['booking status'] = df['booking status'].map(ordinal_mapping)


dummy_columns = df.columns[df.columns.str.contains('room type_|market segment type_|type of meal_')]
for col in dummy_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("\nDataFrame after handling outliers, converting to integers, and encoding:")
print(df.head())


target = 'booking status'
X = df.drop(columns=[target])
y = df[target]


def select_features(X, y, method='chi2', percentile=50):
    if method == 'chi2':
        selector = SelectPercentile(score_func=chi2, percentile=percentile)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices]
    else:
      
        raise ValueError("Unsupported feature selection method.")
    return X_selected, selected_features


X_selected, selected_features = select_features(X, y, method='chi2', percentile=50)

print("\nSelected features using SelectPercentile and chi2:")
print(selected_features)


X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)


param_distributions_knn = {
    'n_neighbors': [int(x) for x in np.linspace(1, 30, num=30)],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


knn = KNeighborsClassifier()


random_search_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_distributions_knn, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)


random_search_knn.fit(X_train, y_train)


print("\nBest Parameters found by RandomizedSearchCV for KNN:")
print(random_search_knn.best_params_)
print("Best Score found by RandomizedSearchCV for KNN:", random_search_knn.best_score_)


best_knn = random_search_knn.best_estimator_

y_pred_knn = best_knn.predict(X_test)


def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy for {model_name}:", accuracy)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix for {model_name}:")
    print(confusion_matrix(y_test, y_pred))

evaluate_model(y_test, y_pred_knn, "KNN")


joblib.dump(best_knn, 'best_knn_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("Model and label encoder saved as best_knn_model.joblib and label_encoder.joblib respectively")


app = Flask(__name__)


model = joblib.load('best_knn_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
   
    features = [
        'number_of_weekend_nights', 'number_of_week_nights', 'car_parking_space', 'lead_time', 
        'repeated', 'average_price', 'special_requests', 'room_type_Room_Type_6', 
        'type_of_meal_Meal_Plan_2', 'market_segment_type_Complementary', 'market_segment_type_Corporate', 
        'market_segment_type_Offline', 'market_segment_type_Online'
    ]
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    
    input_data = pd.DataFrame([data])

    
    input_data = pd.get_dummies(input_data)
    for col in selected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    
    prediction = model.predict(input_data[selected_features])[0]

    
    result = 'Canceled' if prediction == 1 else 'Not Canceled'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
