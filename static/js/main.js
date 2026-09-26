// Optional progressive enhancement: all figures remain normal links without JavaScript.
(() => {
  'use strict';
  const dialog = document.getElementById('figure-dialog');
  if (!dialog || typeof dialog.showModal !== 'function') return;
  const image = document.getElementById('dialog-image');
  const caption = document.getElementById('dialog-caption');
  let trigger = null;
  document.querySelectorAll('a[data-zoom]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      trigger = link;
      const source = link.querySelector('img');
      image.src = link.href;
      image.alt = source ? source.alt : '';
      caption.textContent = link.closest('figure').querySelector('figcaption').textContent;
      dialog.showModal();
      document.body.classList.add('dialog-open');
    });
  });
  dialog.querySelector('button').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('dialog-open');
    if (trigger) trigger.focus();
  });
})();

(() => {
  'use strict';
  const videos = Array.from(document.querySelectorAll('video'));
  videos.forEach(current => {
    current.addEventListener('play', () => {
      videos.forEach(other => { if (other !== current && !other.paused) other.pause(); });
    });
  });
})();
