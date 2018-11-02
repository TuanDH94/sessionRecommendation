from flask_api import FlaskAPI
import flask
from flask import request
from api_handler import ApiHandler

app = FlaskAPI(__name__)

api_handler = ApiHandler()


@app.route('/example/')
def example():
    return {'hello': 'world'}


@app.route('/recommend/', methods=["GET", "POST"])
def recommend():
    inputs = request.args.getlist('inputs')
    print(inputs)
    output = api_handler.get_recommend_output(inputs=inputs)
    return output


if __name__ == "__main__":
    app.run(debug=True)
