"""Fill the PRL 'Authorship Confirmation' form (pages 1-3 of patrec-authorship-and-formatting.pdf).

    python paper/prl/fill_prl_form.py path/to/patrec-authorship-and-formatting.pdf

Writes paper/prl/authorship_confirmation_prl.pdf: page 1 (authorship confirmation, signature left
blank for the author), page 2 (graphical abstract box) and page 3 (research highlights box).
The highlights are read from highlights.txt and the graphical abstract from figs/graphical_abstract.png.
"""
import io
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
W, H = 595.276, 793.701
TITLE = ('Seed-to-seed variation outweighs lightweight attention gains: a grouped, '
         'repeated cross-validation study on field durian disease images')
AUTHOR = 'Lin Ding Shan'
DATE = '3 October 2026'


def y(top):
    """pdfplumber 'top' (from page top) to a reportlab baseline for 10-pt text."""
    return H - top - 8


def wrap(c, text, font, size, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if c.stringWidth(t, font, size) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def page1():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    c.setFont('Times-Roman', 10)
    c.drawCentredString((151.8 + 350.2) / 2, y(223.3), AUTHOR)          # name line
    c.drawCentredString((300.4 + 413.8) / 2, y(410.6), DATE)            # date line
    c.drawString(42.5, y(480), 'None.')
    c.drawString(42.5, y(576), 'None. The work has not been presented at or submitted to any conference.')
    c.drawString(42.5, y(648), 'Not applicable.')
    c.save()
    return buf.getvalue()


def page2():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    c.setFillColorRGB(1, 1, 1)
    c.rect(43.3, H - 625.0, 502.4, 625.0 - 205.4, stroke=0, fill=1)   # clear the template box interior
    c.setFillColorRGB(0, 0, 0)
    x0, width = 54.4, 480
    yy = y(208.4)
    c.setFont('Times-Bold', 10)
    for line in wrap(c, TITLE, 'Times-Bold', 10, width):
        c.drawString(x0, yy, line)
        yy -= 12
    c.setFont('Times-Roman', 10)
    c.drawString(x0, yy, AUTHOR)
    yy -= 10
    img = ImageReader(str(HERE / 'figs' / 'graphical_abstract.png'))
    iw, ih = img.getSize()
    h_img = width * ih / iw
    yy -= h_img
    c.drawImage(img, x0, yy, width=width, height=h_img)
    yy -= 16
    text = ('EfficientNet-B0 was trained on 550 field images of four durian disease classes with and without three '
            'lightweight attention modules (LFA, SE, CBAM) on 20 capture-grouped folds with two seeds each, and compared '
            'in pairs with the corrected resampled t-test. No module met a decision rule fixed before the runs (LFA: '
            '-0.13 percentage points, 95% CI -4.56 to 4.31), while changing only the random seed moved macro F1 by 5.3 '
            'points on average. Single-run gains of a few points on data of this size are not evidence about the architecture.')
    text = text.replace('-0.13', '−0.13').replace('-4.56', '−4.56')
    for line in wrap(c, text, 'Times-Roman', 10, width):
        c.drawString(x0, yy, line)
        yy -= 12
    c.save()
    return buf.getvalue()


def page3():
    hl = [h.strip() for h in (HERE / 'highlights.txt').read_text().splitlines() if h.strip()]
    hl = [h.replace(' -0.13', ' −0.13').replace(' -4.56', ' −4.56') for h in hl]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    c.setFont('Times-Roman', 10)
    for top, text in zip([375.7, 395.6, 415.6, 435.5, 455.4], hl):
        c.drawString(72.0, H - top - 9.3, text)
    c.save()
    return buf.getvalue()


def main(src):
    reader = PdfReader(src)
    writer = PdfWriter()
    for i, overlay in enumerate([page1(), page2(), page3()]):
        page = reader.pages[i]
        page.merge_page(PdfReader(io.BytesIO(overlay)).pages[0])
        writer.add_page(page)
    out = HERE / 'authorship_confirmation_prl.pdf'
    with open(out, 'wb') as f:
        writer.write(f)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1])
