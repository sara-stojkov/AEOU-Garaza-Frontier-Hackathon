chrome.runtime.sendMessage("check_permission", (permitted) => {
  if (!permitted) {
    console.log("No permission - the extension will not run.");
    return;
  }

  console.log("Permission granted — the extension is monitoring the Google search field.");
  function watchSearchInput() {
    const input = document.querySelector('input[name="q"]');
    if (!input) return false;

    console.log("Search input found.");
    console.log("Current input value:", input.value);

    // Track changes as the user types
    input.addEventListener("input", () => {
      console.log("User typed:", input.value);
    });

    // Optional: MutationObserver for autocomplete or dynamic changes
    const observer = new MutationObserver(() => {
      console.log("Vrednost inputa se promenila (MutationObserver):", input.value);
    });

    observer.observe(input, { attributes: true, attributeFilter: ['value'] });

    return true;
  }

  // Trying to find the input immediately
  if (!watchSearchInput()) {
    // If the input is not yet loaded, observe the DOM for changes
    const bodyObserver = new MutationObserver(() => {
      if (watchSearchInput()) {
        // Once found, stop observing the body
        bodyObserver.disconnect();
      }
    });
    bodyObserver.observe(document.body, { childList: true, subtree: true });
  }
});
