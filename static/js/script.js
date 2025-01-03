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

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Response received:", data); // Debug response

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
