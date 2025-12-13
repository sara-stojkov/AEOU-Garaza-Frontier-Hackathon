// content.js

// Debounce function
function debounce(func, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => func.apply(this, args), delay);
  };
}

// Function to attach event listener to Google search input
function attachListener(input) {
  console.log("Google search input detected:", input);

  // Debounced function to send to server
  const sendToServer = debounce((value) => {
    if (!value) return;

    fetch("http://localhost:5000/save_search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: value, timestamp: Date.now() })
    })
    .then(() => console.log("Search sent to server:", value))
    .catch(err => console.error("Error sending to server:", err));
  }, 500);

  // Listen for user typing
  input.addEventListener("input", () => {
    const value = input.value.trim();
    console.log("User typed:", value);
    sendToServer(value);
  });
}

// Function to find the input and attach listener
function watchSearchInput() {
  const input = document.querySelector('input[name="q"]');
  if (input) {
    attachListener(input);
    return true;
  }
  return false;
}

// Initialize
if (!watchSearchInput()) {
  // Observe DOM in case input is added later (Google is dynamic)
  const observer = new MutationObserver(() => {
    if (watchSearchInput()) {
      observer.disconnect();
    }
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });
}

// Optional: detect SPA-style navigation (URL changes without full reload)
let lastUrl = location.href;
setInterval(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    console.log("URL changed, reinitializing input watcher");
    watchSearchInput();
  }
}, 1000);
