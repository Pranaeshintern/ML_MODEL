"""Use an available system font without shipping proprietary font files."""
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def fonts():
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            if "Wualt" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("Wualt", regular))
                pdfmetrics.registerFont(TTFont("WualtBold", bold))
            return "Wualt", "WualtBold"
    return "Helvetica", "Helvetica-Bold"
