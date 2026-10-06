"""Split the verified two-dataset RQ1 graphic into complete vertical panels."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SOURCE = (
    ROOT
    / "HRCAM-major-revision"
    / "materials"
    / "RQ1"
    / "fig_rq1_1_hrcam_group_divergence_2datasets.png"
)
OUTPUT_DIR = ROOT / "HRCAM-major-revision" / "materials" / "RQ1"
ROC_SOURCE = OUTPUT_DIR / "fig_rq1_2_roc_curves_2datasets.png"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def centered_text(draw: ImageDraw.ImageDraw, box, text, text_font, fill):
    x0, y0, x1, y1 = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=text_font,
        fill=fill,
    )


def add_common_x_label(panel: Image.Image):
    draw = ImageDraw.Draw(panel)
    centered_text(
        draw,
        (260, 563, 610, 595),
        "Prediction consistency group",
        font("arial.ttf", 16),
        (45, 45, 45),
    )


def make_panels():
    source = Image.open(SOURCE).convert("RGB")
    if source.size != (1500, 650):
        raise ValueError(f"Unexpected source size: {source.size}")

    # Remove the generic dataset titles and the former shared x-axis label.
    left_crop = source.crop((0, 55, 770, 600))
    right_crop = source.crop((770, 55, 1500, 600))

    left_panel = Image.new("RGB", (820, 600), "white")
    left_panel.paste(left_crop, (25, 0))
    add_common_x_label(left_panel)

    right_panel = Image.new("RGB", (820, 600), "white")
    right_panel.paste(right_crop, (80, 0))
    draw = ImageDraw.Draw(right_panel)
    axis = (45, 45, 45)
    tick_font = font("arial.ttf", 12)
    for value, y_pos in zip(
        [0.18, 0.15, 0.12, 0.09, 0.06, 0.03, 0.00],
        [15, 90, 165, 240, 315, 390, 465],
    ):
        draw.text((48, y_pos - 8), f"{value:.2f}", font=tick_font, fill=axis)

    ylabel = Image.new("RGBA", (170, 30), (255, 255, 255, 0))
    ylabel_draw = ImageDraw.Draw(ylabel)
    ylabel_draw.text((0, 3), "H-RCAM score", font=font("arial.ttf", 16), fill=axis)
    rotated = ylabel.rotate(90, expand=True)
    right_panel.paste(rotated, (10, 180), rotated)
    add_common_x_label(right_panel)

    left_panel.save(
        OUTPUT_DIR / "fig_rq1_1a_hrcam_group_divergence_oxford_pet.png"
    )
    right_panel.save(
        OUTPUT_DIR / "fig_rq1_1b_hrcam_group_divergence_cub200.png"
    )


def make_named_roc_figure():
    """Replace generic dataset headings while preserving the agreed ROC plot."""
    image = Image.open(ROC_SOURCE).convert("RGB")
    if image.size != (1600, 720):
        raise ValueError(f"Unexpected ROC figure size: {image.size}")
    draw = ImageDraw.Draw(image)
    heading_font = font("arialbd.ttf", 18)
    draw.rectangle((250, 18, 550, 62), fill="white")
    draw.rectangle((1000, 18, 1325, 62), fill="white")
    centered_text(
        draw,
        (250, 18, 550, 62),
        "Oxford-IIIT Pet",
        heading_font,
        (45, 45, 45),
    )
    centered_text(
        draw,
        (1000, 18, 1325, 62),
        "CUB-200-2011",
        heading_font,
        (45, 45, 45),
    )
    image.save(OUTPUT_DIR / "fig_rq1_2_roc_curves_named.png")


if __name__ == "__main__":
    make_panels()
    make_named_roc_figure()
