/* Declarative visual assets: no per-lecture library setup is needed. */
const deckURL = new URL('./', location.href);

function localAsset(value) {
  const url = new URL(value, deckURL);
  if (url.origin !== deckURL.origin || !url.pathname.startsWith(deckURL.pathname)) {
    throw new Error(`Store lecture visual assets in the lecture folder: ${value}`);
  }
  return url;
}

export async function initialize(reveal) {
  const plots = [...document.querySelectorAll('[data-plotly]')];
  if (plots.length) {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = new URL('../vendor/plotly/plotly.min.js', import.meta.url).href;
      script.onload = resolve;
      script.onerror = () => reject(new Error('The local Plotly library could not be loaded.'));
      document.head.append(script);
    });
    for (const element of plots) {
      const response = await fetch(localAsset(element.dataset.plotly));
      if (!response.ok) throw new Error(`Missing chart data: ${element.dataset.plotly}`);
      const figure = await response.json();
      const layout = {
        paper_bgcolor: '#fbfbf9', plot_bgcolor: '#fbfbf9',
        font: { family: 'Arial, sans-serif', size: 24, color: '#202b38' },
        margin: { l: 90, r: 35, t: 25, b: 80 },
        colorway: ['#20578c', '#28673f', '#a85a29'],
        ...figure.layout,
        width: 1152, height: 410
      };
      await Plotly.newPlot(element, figure.data, layout, {
        displaylogo: false, responsive: false, scrollZoom: false,
        ...figure.config
      });
    }
  }
  for (const diagram of document.querySelectorAll('img[data-excalidraw-source]')) {
    localAsset(diagram.getAttribute('src'));
    localAsset(diagram.dataset.excalidrawSource);
    await diagram.decode();
  }
  for (const video of document.querySelectorAll('video.animation, video[data-manim-source]')) {
    if (!video.getAttribute('src')) throw new Error('Give every animation a local video source.');
    localAsset(video.getAttribute('src'));
    if (video.hasAttribute('data-manim-source')) localAsset(video.dataset.manimSource);
    if (!video.getAttribute('poster')) throw new Error('Give every animation a poster image for print and before playback.');
    const poster = document.createElement('img');
    poster.src = localAsset(video.getAttribute('poster'));
    poster.alt = video.getAttribute('aria-label') || 'Animation summary';
    poster.className = 'animation-poster';
    video.after(poster);
    await poster.decode();
  }
  reveal.on('slidechanged', ({ previousSlide }) => {
    previousSlide?.querySelectorAll('video').forEach(video => video.pause());
  });
}
