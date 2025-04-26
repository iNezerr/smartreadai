/* Importing Google fonts - Inter */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;900&family=Roboto:wght@100..900&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: "Inter", sans-serif;
}
body {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background-color:rgb(214, 231, 245);
}

.chatbot-popup {
    position: relative;
    width: 450px;
    background: #fff;
    overflow: hidden;
    border-radius: 15px;
    box-shadow: 0 0 128px 0 rgba(0, 0, 0, 0.1), 0 32px 64px -48px rgba(0, 0, 0, 0.5);
}

.chat-header {
    display: flex;
    align-items: center;
    background: #5350c4;
    padding: 15px 22px;
    justify-content: space-between;
}

.chat-header .header-info {
    display: flex;
    gap: 10px;
    align-items: center;
}

.header-info .chatbot-logo {
    height: 35px;
    width: 35px;
    padding: 6px;
    fill: #5350c4;
    flex-shrink: 0;
    background: #fff;
    border-radius: 50%;
}

.header-info .logo-text {
    color: #fff;
    font-size: 1.31rem;
    font-weight: 600;
}

.chat-header #close-chatbot {
    border: none;
    color: #fff;
    height: 40px;
    width: 40px;
    font-size: 1.9rem;
    margin-right: -10px;
    padding-top: 2px;
    cursor: pointer;
    border-radius: 50%;
    background: none;
    transition: 0.2s ease;
}
.chat-header #close-chatbot:hover {
    background: #3d39ac;
}

.chatbot-body {
    padding: 25px 22px;
    display: flex;
    gap: 20px;
    height: 460px;
    margin-bottom: 82px;
    overflow-y: auto;
    flex-direction: column;
}

.chatbot-body .message {
    display: flex;
    gap: 11px;
    align-items: center;
}

.chatbot-body .message .bot-avatar {
    height: 35px;
    width: 35px;
    padding: 6px;
    fill: #fff;
    align-self: flex-end;
    flex-shrink: 0;
    margin-bottom: 2px;
    background: #5350c4;
    border-radius: 50%;  
}

.chatbot-body .user-message {
    flex-direction: column;
    align-items: flex-end;
}

.chatbot-body .message .message-text {
    padding: 12px 16px;
    max-width: 75%;
    font-size: 0.95rem;
    background: #f2f2ff;
}

.chatbot-body .bot-message.thinking .message-text{
    padding: 2px 16px;
}

.chatbot-body .bot-message .message-text {
    background: #f2f2ff;
    border-radius: 13px 13px 13px 3px;
}

.chatbot-body .user-message .message-text {
    color: #fff;
    background: #5350c4;
    border-radius: 13px 13px 3px 13px;
}

.chatbot-body .bot-message .thinking-indicator {
    display: flex;
    gap: 4px;
    padding-block: 15px;
}

.chatbot-body .bot-message .thinking-indicator .dot {
    height: 7px;
    width: 7px;
    opacity: 0.7;
    border-radius: 50%;
    background: #6f6bc2;
    animation: dotpulse 1.8s ease-in-out infinite;
}

.chatbot-body .bot-message .thinking-indicator .dot:nth-child(1) {
    animation-delay: 0.2s;
}
.chatbot-body .bot-message .thinking-indicator .dot:nth-child(1) {
    animation-delay: 0.3s;
}
.chatbot-body .bot-message .thinking-indicator .dot:nth-child(1) {
    animation-delay: 0.4s;
}

@keyframes dotpulse {
    0% 44% {
        transform: translateY(0);
    }
    28% {
        opacity: 0.4;
        transform: translateY(-4px);
    }
    44% {
        opacity: 0.2;
    }
}

.chat-footer {
    position: absolute;
    bottom: 0;
    width: 100%;
    background: #fff;
    padding: 12px 22px 20px;
}
.chat-footer .chat-form {
    display: flex;
    align-items: center;
    border-radius: 32px;
    outline: 1px solid #cccce5;
}

.chat-footer .chat-form:focus-within {
    outline: 2px solid #5350c4;
}

.chat-form .message-input {
    border: none;
    outline: none;
    height: 47px;
    width: 100%;
    resize: none;
    font-size: 0.95rem;
    padding: 4px 0 13px 18px;
    border-radius: inherit;
}
.chat-form .chatbot-controls {
    display: flex;
    height: 47px;
    gap: 3px;
    align-items: center;
    align-self: flex-end;
    padding-right: 6px;
}
.chat-form .chatbot-controls button {
    height: 35px;
    width: 35px;
    border: none;
    font-size: 1.15rem;
    cursor: pointer;
    color: #706db0;
    background: none;
    border-radius: 50%;
    transition: 0.2s ease;
}

.chat-form .chatbot-controls #send-message {
    color: #fff;
    display: none;
    background: #5350c4;
}

.chat-form .message-input:valid~.chatbot-controls #send-message {
    display: block;
}

.chat-form .chatbot-controls #send-message:hover {
    background: #3d39ac;
}

.chat-form .chatbot-controls button:hover {
    background: #f1f1ff;
}
