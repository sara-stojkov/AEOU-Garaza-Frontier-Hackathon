document.getElementById("allow").addEventListener("click", async () => {
  const granted = await chrome.permissions.request({
    origins: ["https://www.google.com/*"]
  });

  document.getElementById("status").innerText = granted
    ? "Permission granted."
    : "Permission denied.";
});
