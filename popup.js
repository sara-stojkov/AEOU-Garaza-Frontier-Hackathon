document.getElementById("allow").addEventListener("click", async () => {
  const granted = await chrome.permissions.request({
    origins: ["https://www.google.com/*"]
  });

  document.getElementById("status").innerText = granted
    ? "Dozvola odobrena."
    : "Dozvola odbijena.";
});
