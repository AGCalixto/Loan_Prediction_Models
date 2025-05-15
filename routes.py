from flask import Blueprint, render_template, request, current_app

routes_blueprint = Blueprint('routes', __name__)


@routes_blueprint.route('/')
def home():
    return render_template('index.html')


@routes_blueprint.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form (index.html)
    form = request.form

    # Convert form values to numeric inputs
    data = [[
        float(form['no_of_dependents']),
        float(form['income_annum']),
        float(form['loan_amount']),
        float(form['loan_term']),
        float(form['cibil_score']),
        float(form['residential_assets_value']),
        float(form['commercial_assets_value']),
        float(form['luxury_assets_value']),
        float(form['bank_asset_value']),
        1 if form['education'] == 'Not Graduate' else 0,
        1 if form['self_employed'] == 'Yes' else 0
    ]]

    model = current_app.config['MODEL']
    prediction = model.predict(data)
    result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'

    return render_template('index.html', result=result)
