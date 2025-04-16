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

  // Set new chat (upload) view as default
  show_view(".new-chat-view");

  // New chat button redirects to upload view
  document.querySelector(".new-chat").addEventListener("click", function() {
      show_view(".new-chat-view");
  });

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
              <p>${text.replace(/\n/g, '<br>')}</p>
          </div>
      `;
      chatContainer.appendChild(messageDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll to latest message
  }

  // Advanced file upload handling
  const fileInput = document.getElementById("fileInput");
  const dropZone = document.getElementById("dropZone");
  const uploadProgress = document.querySelector(".upload-progress");
  const progressBar = document.querySelector(".progress");
  const progressText = document.querySelector(".progress-text");
  const uploadZone = document.querySelector(".upload-zone");
  const uploadBtn = document.querySelector(".upload-btn");
  const browseBtn = document.querySelector(".browse-btn");

  // Handle file browse button
  browseBtn.addEventListener("click", function(e) {
      e.preventDefault();
      fileInput.click();
  });

  // Handle sidebar upload button
  uploadBtn.addEventListener("click", function() {
      show_view(".new-chat-view");
      setTimeout(() => fileInput.click(), 300);
  });

  // Handle drag and drop events
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, highlight, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, unhighlight, false);
  });

  function highlight() {
      dropZone.classList.add('drag-over');
  }

  function unhighlight() {
      dropZone.classList.remove('drag-over');
  }

  dropZone.addEventListener('drop', handleDrop, false);

  function handleDrop(e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length) {
          fileInput.files = files;
          handleFiles(files);
      }
  }

  fileInput.addEventListener("change", function() {
      if (fileInput.files.length) {
          handleFiles(fileInput.files);
      }
  });

  function handleFiles(files) {
      const file = files[0]; // Only handle the first file for now
      
      // Validate file type
      const fileExt = file.name.split(".").pop().toLowerCase();
      if (fileExt !== "pdf" && fileExt !== "epub") {
          alert("Please upload a PDF or EPUB file.");
          return;
      }

      // Show progress
      uploadZone.style.display = "none";
      uploadProgress.style.display = "block";
      
      // Create FormData and upload
      let formData = new FormData();
      formData.append("file", file);

      // Simulated progress (in a real app, you'd use XHR or fetch with progress events)
      let progress = 0;
      const interval = setInterval(() => {
          progress += 5;
          progressBar.style.width = `${Math.min(progress, 90)}%`;
          progressText.textContent = `Processing ${file.name}...`;
          if (progress >= 90) clearInterval(interval);
      }, 200);

      fetch("/upload", {
          method: "POST",
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          clearInterval(interval);
          
          if (data.error) {
              progressText.textContent = data.error;
              progressBar.style.width = "0%";
              setTimeout(() => {
                  uploadZone.style.display = "block";
                  uploadProgress.style.display = "none";
              }, 3000);
          } else {
              progressBar.style.width = "100%";
              progressText.textContent = "File processed successfully!";
              
              // Add the book to conversation list
              addBookToConversations(file.name);
              
              // Switch to chat view after a delay
              setTimeout(() => {
                  show_view(".conversation-view");
                  
                  // Display welcome message
                  displayMessage("assistant", `I've processed your book "${file.name}". What would you like to know about it?`);
              }, 1500);
          }
      })
      .catch(error => {
          clearInterval(interval);
          progressText.textContent = "An error occurred during upload";
          progressBar.style.width = "0%";
          
          setTimeout(() => {
              uploadZone.style.display = "block";
              uploadProgress.style.display = "none";
          }, 3000);
      });
  }

  function addBookToConversations(fileName) {
      const conversationsList = document.querySelector(".conversations");
      const listItem = document.createElement("li");
      listItem.innerHTML = `
          <button>
              <i class="fa fa-book"></i>${fileName.length > 20 ? fileName.substring(0, 20) + '...' : fileName}
          </button>
          <div class="edit-buttons">
              <button><i class="fa fa-pen-to-square"></i></button>
              <button><i class="fa fa-trash"></i></button>
          </div>
          <div class="fade"></div>
      `;
      
      // Add after the "Recent Conversations" grouping
      if (conversationsList.querySelector(".grouping")) {
          conversationsList.insertBefore(listItem, conversationsList.querySelector(".grouping").nextSibling);
      } else {
          conversationsList.appendChild(listItem);
      }
      
      // Add click handler to open this conversation
      listItem.querySelector("button").addEventListener("click", function() {
          show_view(".conversation-view");
      });
  }
});
