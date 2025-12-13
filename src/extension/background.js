chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg === "check_permission") {
    chrome.permissions.contains(
      { origins: ["https://www.google.com/*"] },
      (result) => sendResponse(result)
    );
    return true;
  }
});
