(function() {
  let lastSend = Date.now();
  let lastActivity = Date.now();
  const MAX_ACTIVE_SECONDS = 120;
  const IDLE_THRESHOLD_MS = 2 * 60 * 1000;

  function isActive() {
    return document.visibilityState === 'visible' &&
      (Date.now() - lastActivity) < IDLE_THRESHOLD_MS;
  }

  function sendUsage() {
    if (!isActive()) {
      lastSend = Date.now();
      return;
    }
    const now = Date.now();
    const seconds = Math.min(MAX_ACTIVE_SECONDS, Math.floor((now - lastSend) / 1000));
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

  function onActivity() { lastActivity = Date.now(); }

  document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'hidden') lastSend = Date.now();
  });
  document.addEventListener('mousemove', onActivity);
  document.addEventListener('keydown', onActivity);
  document.addEventListener('scroll', onActivity);
  document.addEventListener('click', onActivity);

  setInterval(sendUsage, 60000);
  window.addEventListener('beforeunload', sendUsage);
})();
