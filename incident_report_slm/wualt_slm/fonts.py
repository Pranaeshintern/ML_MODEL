"""Use system Latin fonts and licensed bundled Devanagari/Tamil fonts."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from .settings import ASSET_ROOT


def script_text(value):
    """Escape source text and explicitly select bundled Indic script fonts."""
    result=escape(str(value if value is not None else 'Not supplied'))
    for pattern, name in [(r'[\u0900-\u097f]+','WualtDevanagari'),(r'[\u0b80-\u0bff]+','WualtTamil')]:
        result=re.sub(pattern,lambda m:f'<font name="{name}">'+m.group()+'</font>',result)
    return result


def fonts():
    for name,filename in [('WualtDevanagari','NotoSansDevanagari-Regular.ttf'),('WualtTamil','NotoSansTamil-Regular.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name,str(ASSET_ROOT/'fonts'/filename),shapable=True))
            pdfmetrics.registerFontFamily(name,normal=name,bold=name,italic=name,boldItalic=name)
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            if "Wualt" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("Wualt", regular,shapable=False))
                pdfmetrics.registerFont(TTFont("WualtBold", bold,shapable=False))
            return "Wualt", "WualtBold"
    return "Helvetica", "Helvetica-Bold"
