from flask import Flask, request, jsonify

from evaluate.evaluate import Evaluate

# from evaluate import Evaluate

app = Flask(__name__)


@app.route('/evaluate', methods=['POST'])
def evaluate_system():
    actual = request.json['actual']
    predicted = request.json['predicted']
    k = request.json.get('k', 10)

    evaluator = Evaluate(actual=actual, predicted=predicted, k=k)
    metrics = evaluator.get_metrics()

    return jsonify(metrics)


if __name__ == '__main__':
    app.run(port=5003)
