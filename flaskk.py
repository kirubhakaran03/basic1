from flask import jsonify, request, Flask
import pickle

app = Flask(__name__)
with open("./logistic_regression_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    mark_req = request.get_json()

    try:
        Temperature = float(mark_req.get('Temperature', 0))
        Humidity = float(mark_req.get('Humidity', 0))
        Wind_Speed = float(mark_req.get('Wind_Speed', 0))
        Cloud_Cover = float(mark_req.get('Cloud_Cover', 0))
        Pressure = float(mark_req.get('Pressure', 0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid input data"}), 400

    # Make prediction and convert the result to a list or integer
    result = clf.predict([[Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure]])
    result = result[0] if len(result) == 1 else result.tolist()

    return jsonify({"loan_approval_status": result})

if __name__ == '__main__':
    app.run(debug=True)
