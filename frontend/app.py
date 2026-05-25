import os

from dotenv import load_dotenv
from flask import Flask, render_template

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    return render_template("index.html", backend_url=backend_url)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
