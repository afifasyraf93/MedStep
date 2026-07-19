from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime, timezone, timedelta
SGT = timezone(timedelta(hours=8))


def generate_pdf_report(
    image_bytes: bytes,
    detections: dict,
    report: dict,
    username: str,
    threshold: float = 0.5
) -> bytes:
    """
    Generate a PDF report for a CXR analysis.
    Returns PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)

    styles = getSampleStyleSheet()
    story = []

    # --- Header ---
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=24, textColor=colors.HexColor("#E63946"),
        spaceAfter=6, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, textColor=colors.grey,
        spaceAfter=4, alignment=TA_CENTER
    )

    story.append(Paragraph("🫁 MedStep", title_style))
    story.append(Paragraph("AI-Assisted Chest X-Ray Analysis Report", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now(SGT).strftime('%Y-%m-%d %H:%M')} (GMT+8)", subtitle_style))
    story.append(Paragraph(f"Student: {username}", subtitle_style))
    story.append(Spacer(1, 0.3 * inch))

    # --- CXR Image ---
    # TODO: convert image_bytes to RLImage
    img_buffer = io.BytesIO(image_bytes)
    rl_img = RLImage(img_buffer, width=3*inch, height=3*inch)
    story.append(rl_img)
    story.append(Spacer(1, 0.2*inch))

    # --- Detection Results Table ---
    story.append(Paragraph("Detection Results", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # TODO: build table data
    # header row: ["Pathology", "Probability", "Severity", "Status"]
    # data rows: loop through detections
    # hint: use get_severity logic — High/Moderate/Low
    # detected = "Detected" if prob >= threshold else "Not Detected"
    table_data = [["Pathology", "Probability", "Severity", "Status"]]
    for pathology, prob in detections.items():
        if prob >= 0.8:
            severity = "High"
        elif prob >= 0.5:
            severity = "Moderate"
        else:
            severity = "Low"
        status = "Detected" if prob >= threshold else "Not Detected"
        # TODO: append row to table_data
        # hint: [pathology.replace("_", " ").title(), f"{round(prob*100,1)}%", severity, status]
        table_data.append([pathology.replace("_", " ").title(), f"{round(prob*100,1)}%", severity, status])

    table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#E63946")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    # --- AI Report ---
    story.append(Paragraph("AI Findings", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))

    # TODO: add findings and impression paragraphs
    # report dict has keys: "findings", "impression"
    # use styles["Normal"] for body text
    # hint: Paragraph(report.get("findings", ""), styles["Normal"])
    story.append(Paragraph(report.get("findings", ""), styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Impression", styles["Heading3"]))
    story.append(Paragraph(report.get("impression", ""), styles["Normal"]))

    story.append(Spacer(1, 0.2 * inch))

    # --- Footer ---
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        "This report is AI-generated for educational purposes only. "
        "Not a substitute for professional medical diagnosis.",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()