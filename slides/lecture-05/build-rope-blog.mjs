// Original classroom drawings accompanying the EleutherAI RoPE reading.
// SVG and editable Excalidraw scenes share the same geometry and labels.
import { writeFile } from 'node:fs/promises';
import { frequencies, rotatePair, toyFrequency } from './position-math.js';

const assets = new URL('./assets/', import.meta.url);
const color = { ink: '#142e4b', muted: '#526374', grid: '#bccbd7',
  purple: '#7550b6', teal: '#147d73' };
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;');

function drawing(name, width, height, description) {
  const svg = [], elements = [];
  function element(type, x, y, w, h, extra = {}) {
    const serial = elements.length + 1;
    const item = { type, id: `${name}-${serial}`, x, y, width: w, height: h,
      angle: 0, strokeColor: color.ink, backgroundColor: 'transparent',
      fillStyle: 'solid', strokeWidth: 2, strokeStyle: 'solid', roughness: 0,
      opacity: 100, groupIds: [], frameId: null, roundness: null,
      seed: serial, version: 1, versionNonce: serial, isDeleted: false,
      boundElements: null, updated: 1791936000000, link: null, locked: false, ...extra };
    elements.push(item);
    return item;
  }
  return {
    text(x, y, text, fill = color.ink, size = 28) {
      svg.push(`<text x="${x}" y="${y}" text-anchor="middle" fill="${fill}" font-size="${size}">${escape(text)}</text>`);
      const w = text.length * size * .55;
      element('text', x - w / 2, y - size, w, size * 1.25, { text,
        originalText: text, fontSize: size, fontFamily: 2, textAlign: 'center',
        verticalAlign: 'top', containerId: null, autoResize: true, lineHeight: 1.25,
        strokeColor: fill });
    },
    line(x1, y1, x2, y2, stroke = color.grid, arrow = false, dashed = false) {
      svg.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${arrow ? 4 : 2}"${dashed ? ' stroke-dasharray="6 5"' : ''}${arrow ? ` marker-end="url(#${name}-${stroke.slice(1)})"` : ''}/>`);
      element(arrow ? 'arrow' : 'line', x1, y1, Math.abs(x2-x1), Math.abs(y2-y1), {
        points: [[0,0],[x2-x1,y2-y1]], strokeColor: stroke, strokeWidth: arrow ? 4 : 2,
        strokeStyle: dashed ? 'dashed' : 'solid', startBinding: null, endBinding: null,
        startArrowhead: null, endArrowhead: arrow ? 'arrow' : null,
      });
    },
    circle(cx, cy, radius) {
      svg.push(`<circle cx="${cx}" cy="${cy}" r="${radius}" fill="none" stroke="${color.grid}" stroke-width="2"/>`);
      element('ellipse', cx-radius, cy-radius, radius*2, radius*2, { strokeColor: color.grid });
    },
    ray(cx, cy, radius, angle, stroke) {
      const [x,y] = rotatePair([radius,0],angle);
      this.line(cx,cy,cx+x,cy-y,stroke,true);
    },
    async save() {
      const markers = [color.purple,color.teal].map(fill => `<marker id="${name}-${fill.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="${fill}" stroke-width="1.8"/></marker>`).join('');
      await writeFile(new URL(`${name}.svg`,assets), `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="Arial, Helvetica Neue, sans-serif" role="img" aria-labelledby="${name}-description"><desc id="${name}-description">${escape(description)}</desc><defs>${markers}</defs>${svg.join('')}</svg>\n`);
      await writeFile(new URL(`${name}.excalidraw`,assets), JSON.stringify({type:'excalidraw',version:2,source:'https://excalidraw.com',elements,appState:{viewBackgroundColor:'#ffffff',gridSize:null},files:{}},null,2)+'\n');
    },
  };
}

const plane = drawing('rope-complex-plane',540,410,
  'A pair of real coordinates is represented on a complex plane. The invented query pair for she starts at (1,0) and rotates through 150 degrees at position five with toy frequency pi/6, reaching (-sqrt(3)/2,1/2). Its unit length is unchanged.');
plane.text(270,32,'“she” · position 5',color.ink,30);
plane.circle(270,204,128);
plane.line(75,204,471,204);
plane.line(270,353,270,56);
plane.text(492,212,'Re',color.muted,26);
plane.text(285,63,'Im',color.muted,26);
plane.ray(270,204,128,0,color.teal);
plane.ray(270,204,128,5*toyFrequency,color.purple);
plane.text(427,245,'1 + 0i',color.teal,28);
plane.text(124,109,'−√3/2 + i/2',color.purple,26);
plane.text(270,383,'150° rotation · same length',color.muted,28);
await plane.save();

const phases = drawing('rope-pair-frequencies',1152,358,
  'One head of width eight has four coordinate pairs. Each supplied query and key pair begins at (1,0). With the standard base 10000, frequencies are 1,0.1,0.01,0.001 radians per token. She at position five rotates by 5,0.5,0.05,0.005 radians. Noa at position zero stays at angle zero in every pair.');
const rates = frequencies(8);
for (const [i,theta] of rates.entries()) {
  const cx = 144 + i * 288, cy = 176, radius = 73;
  phases.text(cx,31,`Coordinates ${2*i}–${2*i+1}`,color.ink,28);
  phases.text(cx,75,`θ = ${theta} rad/token`,color.muted,26);
  phases.circle(cx,cy,radius);
  phases.line(cx-radius-15,cy,cx+radius+15,cy);
  phases.line(cx,cy-radius-15,cx,cy+radius+15);
  phases.ray(cx,cy,radius,0,color.teal);
  phases.ray(cx,cy,radius,5*theta,color.purple);
  phases.text(cx,284,`5θ = ${Number((5*theta).toFixed(3))} rad`,color.purple,28);
}
phases.text(145,342,'she at 5',color.purple,28);
phases.text(400,342,'Noa at 0',color.teal,28);
phases.text(850,342,'Supplied pairs: q = k = (1,0)',color.muted,26);
await phases.save();
console.log('Built two RoPE SVGs and their editable Excalidraw sources.');
