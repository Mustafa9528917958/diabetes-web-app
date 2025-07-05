from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare model
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[key]) for key in request.form]
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        result = "❌ Likely to have Diabetes" if prediction == 1 else "✅ Unlikely to have Diabetes"
        return render_template('index.html', result=result)
    except:
        return render_template('index.html', result="⚠️ Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
