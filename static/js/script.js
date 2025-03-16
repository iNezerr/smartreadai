document.addEventListener("DOMContentLoaded", function () {
  // Sidebar toggle
  const sidebar = document.querySelector("#sidebar");
  const hide_sidebar = document.querySelector(".hide-sidebar");

  hide_sidebar.addEventListener("click", function () {
      sidebar.classList.toggle("hidden");
  });

  // User menu dropdown
  const user_menu = document.querySelector(".user-menu ul");
  const show_user_menu = document.querySelector(".user-menu button");

  show_user_menu.addEventListener("click", function () {
      if (user_menu.classList.contains("show")) {
          user_menu.classList.remove("show");
          setTimeout(() => user_menu.classList.remove("show-animate"), 200);
      } else {
          user_menu.classList.add("show-animate");
          setTimeout(() => user_menu.classList.add("show"), 50);
      }
  });

  // Model selection
  document.querySelectorAll(".model-selector button").forEach(model => {
      model.addEventListener("click", function () {
          document.querySelector(".model-selector button.selected")?.classList.remove("selected");
          model.classList.add("selected");
      });
  });

  // Auto-expanding message input
  const message_box = document.querySelector("#message");

  message_box.addEventListener("keyup", function () {
      message_box.style.height = "auto";
      let height = message_box.scrollHeight + 2;
      if (height > 200) height = 200;
      message_box.style.height = height + "px";
  });

  // View switching
  function show_view(view_selector) {
      document.querySelectorAll(".view").forEach(view => {
          view.style.display = "none";
      });
      document.querySelector(view_selector).style.display = "flex";
  }

  // Set conversation view as default
  show_view(".conversation-view");

  // Handle message sending
  document.querySelector(".send-button").addEventListener("click", sendMessage);
  message_box.addEventListener("keypress", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendMessage();
      }
  });

  function sendMessage() {
      let userMessage = message_box.value.trim();
      if (!userMessage) return;

      // Display user message
      displayMessage("user", userMessage);
      message_box.value = "";
      message_box.style.height = "auto";

      // Send message to backend
      fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: userMessage })
      })
      .then(response => response.json())
      .then(data => {
          displayMessage("assistant", data.answer || "I couldn't process your question.");
      })
      .catch(() => {
          displayMessage("assistant", "An error occurred while fetching the response.");
      });
  }

  function displayMessage(sender, text) {
      let chatContainer = document.querySelector(".conversation-view");
      let messageDiv = document.createElement("div");
      messageDiv.classList.add(sender, "message");

      messageDiv.innerHTML = `
          <div class="identity">
              <i class="${sender === 'user' ? 'user-icon' : 'gpt user-icon'}">
                  ${sender === 'user' ? 'U' : 'G'}
              </i>
          </div>
          <div class="content">
              <p>${text}</p>
          </div>
      `;
      chatContainer.appendChild(messageDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll to latest message
  }

  // File upload handling
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.style.display = "none";
  document.body.appendChild(fileInput);

  document.querySelector(".new-chat").addEventListener("click", function () {
      fileInput.click();
  });

  fileInput.addEventListener("change", function () {
      if (fileInput.files.length === 0) return;

      let formData = new FormData();
      formData.append("file", fileInput.files[0]);

      fetch("/upload", {
          method: "POST",
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          alert(data.message || data.error);
      });

      fileInput.value = ""; // Reset input
  });
});
