from flask_api import FlaskAPI
from flask import request
from api_handler import ApiHandler
from form.form import MainForm
from flask import render_template, flash

app = FlaskAPI(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'

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


@app.route('/index/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        print()

    form = MainForm()
    return render_template('form.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
