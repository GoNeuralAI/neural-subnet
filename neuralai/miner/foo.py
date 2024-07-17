import time
import requests

def foo():
    while True:
        print("Running")
        time.sleep(2)
        get_url = "http://localhost:8093/generate_fe"
        params = {"id": 123, "name": "John Doe"}

        response = requests.get(get_url, params=params)
        print(response.status_code)
        
if __name__ == "__main__":
    foo()