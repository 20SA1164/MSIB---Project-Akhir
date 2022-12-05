const popup = document.querySelector('.chat-popup');
const chatBtn = document.querySelector('.chat-btn');
const submitBtn = document.querySelector('.submit');
const chatArea = document.querySelector('.chat-area');
const inputElm = document.querySelector('input');
const chatUrl = '/chatbot';


// Chat button toggler
chatBtn.addEventListener('click', () => {
	popup.classList.toggle('show');
});

function scrollToBottom(){
	document.getElementById("chatId").scrollTop=document.getElementById("chatId").scrollHeight;
}

// send message
submitBtn.addEventListener('click', () => {
	// capture user inputs from input field after clicking submit
	let userInput = inputElm.value;
	// console.log(userInput);

	// add user inputs into chat area to show user
	let temp = `<div class="out-msg">
		<span class="my-msg">${userInput}</span>
	</div>`;


	chatArea.insertAdjacentHTML("beforeend", temp);

	// remove text from chat input after user sent it.
	inputElm.value= '';

	// fetch(chatUrl, {method:"POST", 
	// 	body: JSON.stringify({"proglang":userInput}), 
	// 	headers:{"Content-type":"application/json; charset=UTF-8"}}
		// ).then(function(u){return u.json();}).then(function(json){jdata=json;})

	// var resd = fetch(chatUrl);
	// .then(res => res.json()).then(data =>dataReceived = data)
	// const data = reponse.json();

	fetch(chatUrl, {
		method:"POST", 
		body: JSON.stringify({"msg":userInput}), 
		headers:{"Content-type":"application/json; charset=UTF-8"}}
	).then(res => res.json()).then(function(data){
		let temp2 = `<div class="income-msg">
			<span class="msg">${data.msg}</span>
			</div>`;
		chatArea.insertAdjacentHTML("beforeend", temp2);
		// console.log(data);
		scrollToBottom();
		});




	
});