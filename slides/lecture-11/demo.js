// Original GIFs play only on the active slide. Text, equations and illustrations
// are authored directly in slides.md; no full-page artwork is loaded.
export async function initialize(Reveal) {
  const images = [...document.querySelectorAll('.native-gif')];
  const printing = Reveal.isPrintView();
  function update() {
    const current = printing || Reveal.isOverview() ? null : Reveal.getCurrentSlide();
    for (const image of images) {
      const desired = image.closest('section') === current ? image.dataset.gifSource : image.dataset.poster;
      if (desired && image.getAttribute('src') !== desired) image.src = desired;
    }
  }
  Reveal.on('slidechanged', update);
  Reveal.on('overviewshown', update);
  Reveal.on('overviewhidden', update);
  update();
}
