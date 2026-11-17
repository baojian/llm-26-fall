// Restore the source deck's live media and links above the native slide artwork.
const GIFS = [
  { slide: 31, file: 'image29.gif', label: 'LLaDA masked-text generation animation' },
  { slide: 33, file: 'image31.gif', label: 'Dream arbitrary-order text generation animation' },
  { slide: 34, file: 'image32.gif', label: 'Dream decoding behavior, center example' },
  { slide: 34, file: 'image33.gif', label: 'Dream decoding behavior, left example' },
  { slide: 34, file: 'image34.gif', label: 'Dream decoding behavior, right example' },
];

function sourceSlide(number) {
  return document.getElementById(number === 69 ? 'references' : `source-${String(number).padStart(3, '0')}`);
}

export async function initialize(Reveal) {
  const response = await fetch(new URL('./assets/source-manifest.json', import.meta.url));
  if (!response.ok) throw new Error('Could not load the original PowerPoint slide manifest.');
  const manifest = await response.json();
  const animations = GIFS.map(({ slide, file, label }) => {
    const image = document.createElement('img');
    image.className = `source-overlay source-gif source-${file.replace('.gif', '')}`;
    image.alt = label;
    image.hidden = true;
    image.dataset.gifSource = `assets/media/${file}`;
    sourceSlide(slide).append(image);
    return image;
  });

  // The source has one click-to-appear picture on slide 6. Before that click,
  // the earlier picture occupies this same region; after it, the native export
  // already contains the final appearance. A fade-out fragment restores both.
  const initialDiagram = document.createElement('img');
  initialDiagram.className = 'source-overlay source-initial-diagram fragment fade-out';
  initialDiagram.src = 'assets/media/image3.png';
  initialDiagram.alt = 'Turning random images into raspberry images is hard.';
  initialDiagram.dataset.fragmentIndex = '0';
  sourceSlide(6).append(initialDiagram);

  for (const [index, slide] of manifest.slides.entries()) {
    for (const { uri, rect } of slide.links || []) {
      if (!/^https?:\/\//i.test(uri) || !Array.isArray(rect) || rect.length !== 4
        || rect.some(value => !Number.isFinite(value))) {
        throw new Error(`Invalid source hyperlink on slide ${index + 1}.`);
      }
      const link = document.createElement('a');
      link.className = 'source-overlay source-link';
      link.href = uri;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.setAttribute('aria-label', uri);
      const [left, top, width, height] = rect;
      Object.assign(link.style, {
        left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px`,
      });
      sourceSlide(index + 1).append(link);
    }
  }

  const printing = new URLSearchParams(location.search).has('print-pdf');
  function updateAnimations() {
    const current = printing || Reveal.isOverview() ? null : Reveal.getCurrentSlide();
    for (const image of animations) {
      if (image.parentElement === current) {
        if (!image.hasAttribute('src')) {
          image.onload = () => { image.hidden = false; };
          image.src = image.dataset.gifSource;
        }
      } else {
        image.hidden = true;
        image.onload = null;
        image.removeAttribute('src');
      }
    }
  }
  Reveal.on('slidechanged', updateAnimations);
  Reveal.on('overviewshown', updateAnimations);
  Reveal.on('overviewhidden', updateAnimations);
  Reveal.sync();
  updateAnimations();
}
