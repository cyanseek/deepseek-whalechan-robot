// Bundle the editable sources into a single offline HTML file. No dependencies.
const fs = require('node:fs');
const path = require('node:path');

function build() {
  const read = name => fs.readFileSync(path.join(__dirname, name), 'utf8').replace(/\r\n/g, '\n');
  let html = read('template.html');
  for (const [name, source] of [
    ['ENGINE', read('engine.js')],
    ['VIEW', read('view.js')],
    ['APP', `const STAR_LEVEL = ${JSON.stringify(JSON.parse(read('level.json')))};\n${read('app.js')}`],
  ]) {
    html = html.replace(`<!-- ${name} -->`, () => `<script>\n${source}</script>`);
  }
  return html;
}

if (require.main === module) {
  const destination = path.join(__dirname, 'index.html');
  const html = build();
  if (process.argv.includes('--check')) {
    if (!fs.existsSync(destination) || fs.readFileSync(destination, 'utf8') !== html) {
      console.error('Offline page is stale. Run: node docs/play/build.cjs');
      process.exitCode = 1;
    }
  } else fs.writeFileSync(destination, html);
}
module.exports = build;
