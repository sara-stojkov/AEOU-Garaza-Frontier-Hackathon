chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg === "check_permission") {
    chrome.permissions.contains(
      { origins: ["https://www.google.com/*"] },
      (result) => sendResponse(result)
    );
    return true;
  }
  
  if (msg.action === "getUserInfo") {
    chrome.identity.getProfileUserInfo({ accountStatus: "ANY" }, (userInfo) => {
      if (userInfo && userInfo.email) {
        sendResponse({ email: userInfo.email, id: userInfo.id });
      } else {
        sendResponse({ email: null });
      }
    });
    return true; // Keep the message channel open for async response
  }
});
