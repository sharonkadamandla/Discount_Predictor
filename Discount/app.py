from flask import Flask,render_template,request,jsonify

app=Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_data = request.json.get('input_data')
        processed_data = "working" # input_data is the variable we can use da 
        return jsonify({'processed_data': processed_data})
    else:
        return render_template('home.html')
    
@app.route("/result")
def result():
    return render_template('about.html')
    

if __name__=="__main__":
    app.run(debug=True)