import pytest
from Flask_App import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Loan Approval" in response.data  # Adjust based on your HTML


def test_prediction(client):
    form_data = {
        'no_of_dependents': '1',
        'income_annum': '500000',
        'loan_amount': '100000',
        'loan_term': '12',
        'cibil_score': '700',
        'residential_assets_value': '200000',
        'commercial_assets_value': '100000',
        'luxury_assets_value': '0',
        'bank_asset_value': '100000',
        'education': 'Graduate',
        'self_employed': 'No'
    }
    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b'Approved' in response.data or b'Rejected' in response.data
