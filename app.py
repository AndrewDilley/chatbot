from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Define the '/about' route
@app.route('/about')
def about():
    return "This is the About Page!"


if __name__ == '__main__':
    app.run(debug=True)
