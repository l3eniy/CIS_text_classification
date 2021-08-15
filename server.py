from flask import Flask, json, request, jsonify
from blackbox import blackbox

# sample call
# curl -X POST localhost:5000/classify --data '{"text":"ensure ldap server is not installed"}'  --header "Content-Type: application/json"

api = Flask(__name__)

@api.route('/classify', methods=['POST'])
def classify_text():
    print(request.data)
    if(request.is_json):
        data = request.json
        print(data)
        txt = data['text']
        print(txt)

        bb = blackbox()
        prediction_probabilities = bb.get_prediction_probability_for_sample(txt)
        print(prediction_probabilities)
        obj = bb.get_probability_map(prediction_probabilities)
        return jsonify(obj)
    else:
        return jsonify("FAILED")

if __name__ == '__main__':
    api.run()