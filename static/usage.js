(function() {
  let lastSend = Date.now();
  function sendUsage() {
    const now = Date.now();
    const seconds = Math.floor((now - lastSend) / 1000);
    if (seconds > 0) {
      lastSend = now;
      fetch('/api/usage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seconds }),
        keepalive: true
      }).catch(function() {});
    }
  }
  setInterval(sendUsage, 60000);
  window.addEventListener('beforeunload', sendUsage);
})();
