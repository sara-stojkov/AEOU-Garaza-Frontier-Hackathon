chrome.runtime.sendMessage("check_permission", (permitted) => {
  if (!permitted) {
    console.log("Nema dozvole — ekstenzija ne radi.");
    return;
  }

  console.log("Dozvola postoji — ekstenzija prati Google search polje.");

  function watchSearchInput() {
    const input = document.querySelector('input[name="q"]');
    if (!input) return false;

    console.log("Search input pronađen.");
    console.log("Trenutna vrednost inputa:", input.value);

    // Praćenje promena dok korisnik unosi
    input.addEventListener("input", () => {
      console.log("Korisnik unosi:", input.value);
    });

    // Opcionalno: MutationObserver za autocomplete ili dinamičke promene
    const observer = new MutationObserver(() => {
      console.log("Vrednost inputa se promenila (MutationObserver):", input.value);
    });

    observer.observe(input, { attributes: true, attributeFilter: ['value'] });

    return true;
  }

  // Pokušavamo da nađemo input odmah
  if (!watchSearchInput()) {
    // Ako input još nije učitan, pratimo DOM za promene
    const bodyObserver = new MutationObserver(() => {
      if (watchSearchInput()) {
        // Kada ga pronađemo, prekidamo posmatranje body-a
        bodyObserver.disconnect();
      }
    });
    bodyObserver.observe(document.body, { childList: true, subtree: true });
  }
});
