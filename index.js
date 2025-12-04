document.getElementById('toggleDisplay').addEventListener('click', () => {
  document.body.classList.toggle('show-dialog');
  waterfall.distributeItems();
});
document.getElementById('toggleBubble').addEventListener('click', () => {
  document.body.classList.toggle('wechat-bubble');
  waterfall.distributeItems();
});
document.getElementById('toggleTitle').addEventListener('click', () => {
  document.body.classList.toggle('show-title');
  waterfall.distributeItems();
});
document.getElementById('toggleDetail').addEventListener('click', () => {
  document.body.classList.toggle('show-detail');
  waterfall.distributeItems();
});

const res = await fetch('./assets/quotations.json');
/** @type {{ width: number, height: number, title: string, format: string, messages: string[] }[]} */
const items = await res.json();

const params = new URLSearchParams(window.location.search);

/** @type {HTMLTemplateElement} */
const messageTemplate = document.getElementById('message-template');

function renderCard({ width, height, title, format = 'png', messages, footer }) {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.title = title;

  const container = document.createElement('div');
  container.className = 'card-messages';
  let currentSender, lastSender;
  for (let message of messages) {
    const clone = messageTemplate.content.cloneNode(true);
    if (message.startsWith('@')) {
      clone.querySelector('.message-avatar').src = './assets/default-avatar.png';
      clone.querySelector('.message-name').innerText = '群友';
      message = message.substring(1);
      currentSender = "群友";
    } else if (message.startsWith('#')) {
      clone.querySelector('.message-avatar').src = './assets/default-avatar.png';
      clone.querySelector('.message-avatar').style.filter = 'invert(1)';
      clone.querySelector('.message-name').innerText = '另一位群友';
      message = message.substring(1);
      currentSender = "另一位群友";
    } else {
      currentSender = "静哥哥"
    }
    if (currentSender === lastSender) {
      clone.querySelector('.message').classList.add('message-continue');
    }
    clone.querySelector('.message-bubble').innerHTML = message;
    container.appendChild(clone);
    lastSender = currentSender;
  }
  card.appendChild(container);

  const img = document.createElement('img');
  img.className = 'card-image';
  img.alt = title;
  img.src = `./assets/quotations/${title}.${format}`;
  img.style.aspectRatio = `${width} / ${height}`;
  card.appendChild(img);

  messages = messages.map(message => message.replace(/^[@#](.*)/, '<cite>$1</cite>')).join('<br>');
  card.innerHTML += `\
    <div class="card-content">
      <h2 class="card-title"><a href="#${title}">${title}</a></h2>
      <div class="card-text">${messages}</div>
    </div>`;
  if (footer) {
    card.innerHTML += `<div class="card-footer">${footer}</div>`;
  }
  return card;
}

const waterfall = new Waterfall('waterfall', items.map(item => renderCard(item)), 20);

for (let value of params.getAll('mock')) {
  let [title, ...messages] = value.split(':');
  let card = renderCard({ title, messages }, true);
  card.classList.add('mock');
  waterfall.elements.push(card);
}

function resizeWaterfall() {
  if (window.innerWidth >= 1440) {
    waterfall.handleResize(4);
  } else if (window.innerWidth >= 1024) {
    waterfall.handleResize(3);
  } else if (window.innerWidth >= 768) {
    waterfall.handleResize(2);
  } else {
    waterfall.handleResize(1);
  }
}
window.addEventListener('resize', resizeWaterfall);

function scrollToHashIfValid() {
  const hash = decodeURIComponent(window.location.hash.substring(1));
  const element = document.querySelector(`[data-title="${hash}"]`);
  if (element) {
    element.classList.add('highlight');
    setTimeout(() => element.classList.remove('highlight'), 1500);
    element.scrollIntoView({ behavior: 'smooth' });
  }
}
window.addEventListener('hashchange', scrollToHashIfValid);

resizeWaterfall();
scrollToHashIfValid();
document.getElementById('toggleDisplay').click();
document.getElementById('toggleTitle').click();

if (/Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
  waterfall.handleResize(1);
}