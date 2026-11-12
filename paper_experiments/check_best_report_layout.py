"""Render Word-exported PDF and verify editable-report data coverage."""
import json
from pathlib import Path
import pypdfium2 as pdfium
from docx import Document
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/ia_vpdi_report_20261028'
QA = OUT / 'qa'


def main():
    pdf = pdfium.PdfDocument(QA / 'report.pdf')
    pages = []
    text = []
    for i, page in enumerate(pdf):
        image = page.render(scale=1.4).to_pil().convert('RGB')
        image.save(QA / f'page-{i+1}.png')
        thumb = image.copy()
        thumb.thumbnail((420, 600))
        canvas = Image.new('RGB', (440, 630), '#DDDDDD')
        canvas.paste(thumb, ((440-thumb.width)//2, 20))
        ImageDraw.Draw(canvas).text((10, 605), f'Page {i+1}', fill='black')
        pages.append(canvas)
        text.append(page.get_textpage().get_text_range())
    sheet = Image.new('RGB', (440*3, 630*((len(pages)+2)//3)), '#BBBBBB')
    for i, page in enumerate(pages):
        sheet.paste(page, ((i%3)*440, (i//3)*630))
    sheet.save(QA / 'contact.png')
    doc = Document(OUT / 'IA-VPDI论文实验结果更新报告_20261028.docx')
    checks = dict(pages=len(pdf), editable_tables=len(doc.tables), figures=len(doc.inline_shapes),
                  native_word_pdf_render=True, source_data_status='data_status.json',
                  page_text_characters=[len(s) for s in text])
    (QA / 'layout_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (QA / 'page_text.txt').write_text('\n\n---PAGE---\n\n'.join(text), encoding='utf-8')
    print(checks)


if __name__ == '__main__':
    main()
