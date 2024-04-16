from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("home.html")

# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
