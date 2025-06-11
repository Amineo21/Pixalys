import unittest
from app import create_app

class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_home_page_status_code(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_main_page_status_code(self):
        response = self.client.get('/main')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
