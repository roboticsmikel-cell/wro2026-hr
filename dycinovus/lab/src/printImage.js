// Print one image through the browser - Baybayin, when the backend has no
// printer of its own (Render; /state says server_prints: false).
//
// The image goes into a hidden iframe holding nothing else, so the page that
// prints is the glyphs alone, centred and fitted to the paper, not a
// screenshot of the console. Printing waits for the image to load: printing
// first gives a blank page.
//
// Chrome shows its print dialog; nothing a page can do skips it. To print
// with no dialog, start Chrome with --kiosk-printing, which sends every print
// straight to the default printer.

const escapeAttr = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;')
  .replace(/</g, '&lt;')

export function printImage(url, title = 'Baybayin') {
  const frame = document.createElement('iframe')
  frame.setAttribute('aria-hidden', 'true')
  frame.style.cssText = 'position:fixed;right:0;bottom:0;width:0;height:0;border:0'
  frame.srcdoc = `<!doctype html><html><head><title>${escapeAttr(title)}</title>
<style>
  @page { margin: 10mm; }
  html, body { margin: 0; height: 100%; }
  body { display: flex; align-items: center; justify-content: center; }
  img { max-width: 100%; max-height: 100%; object-fit: contain; }
</style></head><body><img src="${escapeAttr(url)}" alt=""></body></html>`

  let removed = false
  const remove = () => {
    if (!removed) { removed = true; frame.remove() }
  }

  frame.onload = () => {
    const win = frame.contentWindow
    const img = frame.contentDocument?.querySelector('img')
    if (!win || !img) { remove(); return }
    const go = () => {
      win.addEventListener('afterprint', remove)
      win.focus()
      win.print()
      setTimeout(remove, 60000)   // in case afterprint never fires
    }
    if (img.complete && img.naturalWidth) go()
    else {
      img.onload = go
      img.onerror = remove
    }
  }
  document.body.appendChild(frame)
}
