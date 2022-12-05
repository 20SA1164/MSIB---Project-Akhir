const popup = document.querySelector('.chat-popup');
const chatBtn = document.querySelector('.chat-btn');
const submitBtn = document.querySelector('.submit');
const chatArea = document.querySelector('.chat-area');
const inputElm = document.querySelector('input');

const emojiBtn = document.querySelector("#emoji-btn");
const picker = new EmojiButton();


// Emoji selection
window.addEventListener('DOMContentLoaded', () => {
	picker.on('emoji', emoji => {
		document.querySelector('input').value += emoji;
	});

	emojiBtn.addEventListener('click', () => {
		picker.togglePicker(emojiBtn);
	});
});


// Chat button toggler
chatBtn.addEventListener('click', () => {
	popup.classList.toggle('show');
});


// send message
submitBtn.addEventListener('click', () => {
	// capture user inputs from input field after clicking submit
	let userInput = inputElm.value;
	// console.log(userInput);

	// add user inputs into chat area to show user
	let temp = `<div class="out-msg">
		<span class="my-msg">${userInput}</span>
		<img src="img/jerry.jpg" class="avatar">
	</div>`;


	chatArea.insertAdjacentHTML("beforeend", temp);

	// remove text from chat input after user sent it.
	inputElm.value= '';
});