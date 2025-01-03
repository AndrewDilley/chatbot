const chatForm = document.getElementById('chat-form');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

// Add default welcome message on page load
window.onload = () => {
    chatBox.innerHTML += `<div class="ai-message">Hi, I can answer any questions you may have about our policies.</div><br>`;
};

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent form from reloading the page

    const message = userInput.value.trim();
    if (!message) return; // Ignore empty messages

    // Display user message
    chatBox.innerHTML += `<div class="user-message">${message}</div>`;
    userInput.value = ''; // Clear the input field

    // Display AI response
    chatBox.innerHTML += `<div class="ai-message">AI is typing...</div>`;
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await response.json();

        // Replace "AI is typing..." with the actual response and include reference
        chatBox.innerHTML = chatBox.innerHTML.replace('<div class="ai-message">AI is typing...</div>', '');
        chatBox.innerHTML += `<div class="ai-message">${data.response}</div><br>`;
    } catch (error) {
        console.error("Error fetching response:", error); // Debug errors
        chatBox.innerHTML = chatBox.innerHTML.replace('<div class="ai-message">AI is typing...</div>', '');
        chatBox.innerHTML += `<div class="error-message">Error: Unable to fetch AI response.</div>`;
    }

    // Auto-scroll to the latest message
    chatBox.scrollTop = chatBox.scrollHeight;
});
