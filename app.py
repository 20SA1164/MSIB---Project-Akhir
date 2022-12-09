from flask import Flask, render_template, jsonify, request
import random
import json
import joblib
import numpy as np
import torch
from chatbot.nltk_utils import bag_of_words, tokenize
from chatbot.model import NeuralNet

app = Flask(__name__)

try:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	with open('chatbot/intents.json', 'r') as f:
		intents = json.load(f)

	FILE = "chatbot/data.pth"
	data = torch.load(FILE)

	input_size = data["input_size"]
	hidden_size = data["hidden_size"]
	output_size = data["output_size"]
	all_words = data["all_words"]
	tags = data["tags"]
	model_state = data["model_state"]

	model = NeuralNet(input_size, hidden_size, output_size).to(device)
	model.load_state_dict(model_state)
	model.eval()

except Exception as e:
	print(e)


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/chatbot', methods=["POST"])
def chatbot_msg():
	if request.method == "POST":
		user_data = request.json

		sentence = user_data['msg']
	
		sentence = tokenize(sentence)
		X = bag_of_words(sentence, all_words)
		X = X.reshape(1, X.shape[0])
		X = torch.from_numpy(X)

		output = model(X)
		_, predicted = torch.max(output, dim=1)
		tag = tags[predicted.item()]
		probs = torch.softmax(output, dim=1)
		prob = probs[0][predicted.item()]

		if prob.item() > 0.75:
			for intent in intents["intents"]:
				if tag == intent["tag"]:
			
					return jsonify(msg=random.choice(intent['responses']))
		else:
			return jsonify(msg="I do not understand...")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method ==  'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = float(request.form['credit'])
        area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        # gender
        if (gender == "Male"):
            male=1
        else:
            male=0
        
        # married
        if(married=="Yes"):
            married_yes = 1
        else:
            married_yes=0

        # dependents
        if(dependents=='1'):
            dependents_1 = 1
            dependents_2 = 0
            dependents_3 = 0
        elif(dependents == '2'):
            dependents_1 = 0
            dependents_2 = 1
            dependents_3 = 0
        elif(dependents=="3+"):
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 1
        else:
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 0  

        # education
        if (education=="Not Graduate"):
            not_graduate=1
        else:
            not_graduate=0

        # employed
        if (employed == "Yes"):
            employed_yes=1
        else:
            employed_yes=0

        # property area

        if(area=="Semiurban"):
            semiurban=1
            urban=0
        elif(area=="Urban"):
            semiurban=0
            urban=1
        else:
            semiurban=0
            urban=0


        ApplicantIncomelog = np.log(ApplicantIncome)
        totalincomelog = np.log(ApplicantIncome+CoapplicantIncome)
        LoanAmountlog = np.log(LoanAmount)
        Loan_Amount_Termlog = np.log(Loan_Amount_Term)
        model1 = joblib.load(open('model.pkl', 'rb'))
        prediction = model1.predict([[credit, ApplicantIncomelog,LoanAmountlog, Loan_Amount_Termlog, totalincomelog, male, married_yes, dependents_1, dependents_2, dependents_3, not_graduate, employed_yes,semiurban, urban ]])

        # print(prediction)

        if(prediction=="N"):
            prediction="Tidak dapat Mengajukan Pinjaman. Karena dari data yang anda masukkan dengan pertimbangan pengeluaran anda, maka pendapatan anda tidak sesuai dengan jumlah pinjaman yang akan anda ajukan."
        else:
            prediction="Dapat Mengajukan Pinjaman. Karena dari data yang anda masukkan dengan pertimbangan pengeluaran anda, maka pendapatan anda sesuai dengan jumlah pinjaman yang akan anda ajukan"


        return render_template("prediction.html", prediction_text="Hasil Prediksi dari data yang telah diinputkan adalah Anda {}".format(prediction))


    else:
        return render_template("prediction.html")

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0:8080')
