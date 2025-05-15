import joblib
from routes import routes_blueprint
from flask import Flask


def create_app():
    app = Flask(__name__)

    # Load model once and attach to app config
    model = joblib.load('loan_model.pkl')
    app.config['MODEL'] = model

    # Register Blueprint
    app.register_blueprint(routes_blueprint)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
