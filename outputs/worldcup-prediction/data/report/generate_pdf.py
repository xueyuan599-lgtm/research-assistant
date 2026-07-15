#!/usr/bin/env python3
"""
Generate a professional PDF report for the 2026 World Cup Prediction.
Uses reportlab for high-quality output.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import datetime

# ============================================================
# Paths
# ============================================================
BASE = "E:/wuyi/数学建模半自动/research-assistant/outputs/worldcup-prediction/data"
REPORT_DIR = os.path.join(BASE, "report")
FIGURES = {
    'champion': os.path.join(REPORT_DIR, '01_champion_probability_bar.png'),
    'semifinal': os.path.join(REPORT_DIR, '02_semifinal_probability_bar.png'),
    'confederation': os.path.join(REPORT_DIR, '03_champion_probability_by_confederation.png'),
    'bracket': os.path.join(REPORT_DIR, '04_tournament_bracket.png'),
    'features': os.path.join(REPORT_DIR, '05_feature_importance_shap.png'),
    'comparison': os.path.join(REPORT_DIR, '06_model_comparison.png'),
    'elo': os.path.join(REPORT_DIR, '07_elo_ranking_changes.png'),
}
PDF_PATH = os.path.join(REPORT_DIR, 'worldcup_prediction_report_en.pdf')

# ============================================================
# Color palette
# ============================================================
NAVY = HexColor('#1B2A4A')
DARK_BLUE = HexColor('#2C3E50')
MID_BLUE = HexColor('#3498DB')
LIGHT_BLUE = HexColor('#5DADE2')
ORANGE = HexColor('#E67E22')
RED = HexColor('#C0392B')
GREEN = HexColor('#27AE60')
GOLD = HexColor('#D4A843')
LIGHT_GRAY = HexColor('#F5F5F5')
MED_GRAY = HexColor('#95A5A6')
DARK_GRAY = HexColor('#555555')

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm

# ============================================================
# Styles
# ============================================================
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    'ReportTitle', parent=styles['Title'],
    fontName='Helvetica-Bold', fontSize=28, leading=34,
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=4*mm
)

style_subtitle = ParagraphStyle(
    'ReportSubtitle', parent=styles['Normal'],
    fontName='Helvetica', fontSize=13, leading=18,
    textColor=DARK_GRAY, alignment=TA_CENTER, spaceAfter=8*mm
)

style_h1 = ParagraphStyle(
    'H1', parent=styles['Heading1'],
    fontName='Helvetica-Bold', fontSize=20, leading=26,
    textColor=NAVY, spaceBefore=10*mm, spaceAfter=4*mm,
    borderWidth=0, borderPadding=0,
)

style_h2 = ParagraphStyle(
    'H2', parent=styles['Heading2'],
    fontName='Helvetica-Bold', fontSize=15, leading=20,
    textColor=DARK_BLUE, spaceBefore=6*mm, spaceAfter=3*mm,
)

style_h3 = ParagraphStyle(
    'H3', parent=styles['Heading3'],
    fontName='Helvetica-Bold', fontSize=12, leading=16,
    textColor=MID_BLUE, spaceBefore=4*mm, spaceAfter=2*mm,
)

style_body = ParagraphStyle(
    'BodyReport', parent=styles['Normal'],
    fontName='Helvetica', fontSize=10, leading=15,
    textColor=DARK_GRAY, alignment=TA_JUSTIFY,
    spaceBefore=1*mm, spaceAfter=2*mm,
)

style_body_center = ParagraphStyle(
    'BodyCenter', parent=style_body,
    alignment=TA_CENTER, spaceBefore=2*mm, spaceAfter=4*mm,
)

style_caption = ParagraphStyle(
    'Caption', parent=styles['Normal'],
    fontName='Helvetica-Oblique', fontSize=9, leading=12,
    textColor=MED_GRAY, alignment=TA_CENTER,
    spaceBefore=1*mm, spaceAfter=5*mm,
)

style_bullet = ParagraphStyle(
    'Bullet', parent=style_body,
    leftIndent=12, bulletIndent=0,
    spaceBefore=0.5*mm, spaceAfter=0.5*mm,
)

style_footer = ParagraphStyle(
    'Footer', parent=styles['Normal'],
    fontName='Helvetica-Oblique', fontSize=8, leading=10,
    textColor=MED_GRAY, alignment=TA_CENTER,
)

style_code = ParagraphStyle(
    'Code', parent=styles['Code'],
    fontName='Courier', fontSize=8, leading=11,
    textColor=DARK_BLUE, backColor=LIGHT_GRAY,
    borderWidth=0.5, borderColor=MED_GRAY,
    borderPadding=4, spaceBefore=3*mm, spaceAfter=3*mm,
)

# ============================================================
# Helper functions
# ============================================================

def make_table(data, col_widths=None, header=True):
    """Create a styled table from a list-of-lists."""
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), DARK_GRAY),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, MED_GRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    if header and len(data) > 0:
        style_cmds += [
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9.5),
        ]
    # Alternate row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), HexColor('#F8F9FA')))

    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style_cmds))
    return t


def add_figure(path, width=14*cm, max_height=11*cm):
    """Add an image with constrained dimensions, returns None if missing."""
    if os.path.exists(path):
        from PIL import Image as PILImage
        with PILImage.open(path) as pil_img:
            w, h = pil_img.size
        aspect = h / w
        use_w = min(width, 16*cm)
        use_h = use_w * aspect
        if use_h > max_height:
            use_h = max_height
            use_w = use_h / aspect
        img = Image(path, width=use_w, height=use_h)
        return img
    return None


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=MED_GRAY,
                      spaceBefore=3*mm, spaceAfter=3*mm)

# ============================================================
# Build PDF
# ============================================================
doc = SimpleDocTemplate(
    PDF_PATH, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title='2026 World Cup Prediction Report',
    author='LightGBM + Monte Carlo Simulation Pipeline',
)

story = []

# ==================== COVER / TITLE ====================
story.append(Spacer(1, 3*cm))
story.append(Paragraph('2026 World Cup Prediction Report', style_title))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    'France Favorite with 28.19% Champion Probability',
    style_subtitle
))
story.append(Spacer(1, 8*mm))
story.append(hr())
story.append(Spacer(1, 5*mm))

# Meta info table
meta_data = [
    ['Generated', '2026-07-09'],
    ['Model', 'LightGBM (Optuna-optimized, 150 trials)'],
    ['Simulations', '10,000 Monte Carlo runs'],
    ['Features', '72-dimensional match prediction'],
    ['Training Data', '4,622 international matches (2021-2026)'],
    ['Predicted Champion', 'France — 28.19%'],
]
meta_table = Table(meta_data, colWidths=[4*cm, 10*cm])
meta_table.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('TEXTCOLOR', (0, 0), (-1, -1), DARK_GRAY),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('LINEBELOW', (0, 0), (-1, -1), 0.3, HexColor('#E0E0E0')),
]))
story.append(meta_table)

story.append(Spacer(1, 2*cm))

# ==================== SECTION 1: Executive Summary ====================
story.append(Paragraph('1. Executive Summary', style_h1))
story.append(hr())

p = '<b>Predicted Champion: France — 28.19% probability</b>'
story.append(Paragraph(p, style_body))

story.append(Paragraph(
    'France leads all Monte Carlo simulations with the highest championship probability, '
    'driven by a world-class Elo rating (1903), deep tournament experience, and dominant '
    'attacking and defensive metrics. Reigning champion Argentina (20.76%) and Spain '
    '(18.24%) form the second tier of contenders, with the semifinal picture heavily '
    'concentrated among traditional European and South American powers.',
    style_body
))

img = add_figure(FIGURES['champion'])
if img:
    story.append(Spacer(1, 3*mm))
    story.append(img)
    story.append(Paragraph('Figure 1: Top 8 champion probabilities with error bars. France leads at 28.19%.', style_caption))

# ==================== SECTION 2: Data & Methodology ====================
story.append(Paragraph('2. Data &amp; Methodology', style_h1))
story.append(hr())

story.append(Paragraph('2.1 Data Sources', style_h2))
data_src = [
    ['Data', 'Source', 'Time Range'],
    ['Historical matches', 'Elo Rating / FIFA Rankings', '1872–2026'],
    ['World Cup history', 'FIFA Official Data', '1930–2026'],
    ['Recent matches', 'FIFA A-level internationals', '2021–2026'],
    ['Tournament schedule', '2026 World Cup Official', 'Summer 2026'],
]
story.append(make_table(data_src, col_widths=[4*cm, 5*cm, 4*cm]))

story.append(Paragraph('2.2 Model Architecture', style_h2))
story.append(Paragraph(
    'The prediction pipeline consists of four stages:',
    style_body
))
stages = [
    '<b>Feature Engineering:</b> Extract 72-dimensional features from raw match records, '
    'covering Elo ratings, FIFA rankings, recent form windows (5/10 matches), '
    'World Cup historical performance, goal stability, venue factors, and rest days.',
    '<b>Baseline Comparison:</b> Evaluate four methods simultaneously — '
    'Elo probability model, Poisson regression, XGBoost, and LightGBM.',
    '<b>Model Optimization:</b> Select LightGBM as the final model, with 150 Optuna '
    'Bayesian hyperparameter search trials (TPE sampler).',
    '<b>Monte Carlo Simulation:</b> Run 10,000 full knockout simulations starting '
    'from the actual quarterfinal bracket, with probabilistic draws resolved '
    'via weighted penalty shootouts.',
]
for s in stages:
    story.append(Paragraph(f'&bull; {s}', style_bullet))

story.append(Paragraph('2.3 Final Model Parameters', style_h2))
code_text = (
    'num_leaves: 43      max_depth: 14       min_child_samples: 5<br/>'
    'learning_rate: 0.009  n_estimators: 730  subsample: 0.905<br/>'
    'colsample_bytree: 0.706  reg_alpha: 1.9e-5  reg_lambda: 0.026'
)
story.append(Paragraph(code_text, style_code))

story.append(Paragraph('2.4 Evaluation Metrics', style_h2))
metrics_data = [
    ['Metric', 'Training', 'Validation', 'CV (Mean ± SD)'],
    ['Accuracy', '0.8313', '0.6264', '0.6148 ± 0.042'],
    ['LogLoss', '0.5546', '0.8409', '0.8555 ± 0.064'],
    ['Brier Score', '0.1006', '0.1640', '0.1667 ± 0.014'],
    ['AUC (OVR)', '0.9665', '0.7610', '0.7493 ± 0.039'],
]
story.append(make_table(metrics_data, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 4*cm]))

story.append(Paragraph('2.5 Robustness', style_h2))
story.append(Paragraph(
    'Validation performance across three random seeds (42, 123, 2024) is highly '
    'consistent: AUC 0.761–0.762, LogLoss 0.839–0.841. The AUC standard deviation '
    'across seeds is only 0.0008, LogLoss SD is 0.0012, indicating stable training '
    'that is largely insensitive to initialization.',
    style_body
))

# ==================== SECTION 3: Baseline Comparison ====================
story.append(Paragraph('3. Baseline Comparison', style_h1))
story.append(hr())

baseline_data = [
    ['Model', 'Val. Accuracy', 'Val. LogLoss', 'CV AUC', 'Note'],
    ['LightGBM', '0.618', '0.839', '0.730', 'Gradient boosted trees, 72 features'],
    ['XGBoost', '0.560', '0.920', '0.733', 'Comparable tree model'],
    ['Elo', '0.587', '0.915', '0.664', 'Elo rating only, no features'],
    ['Poisson', '0.152', '1.748', '0.345', 'Historical goal rates only'],
]
story.append(make_table(baseline_data, col_widths=[2.5*cm, 2.5*cm, 2.5*cm, 2*cm, 5*cm]))

story.append(Paragraph(
    'LightGBM leads in both accuracy and log loss. The Elo baseline performs reasonably '
    'well without additional features (CV AUC = 0.664), confirming that Elo ratings '
    'capture meaningful predictive signal. The Poisson model underperforms due to its '
    'reliance on raw goal averages without capturing match-level dependencies.',
    style_body
))

img = add_figure(FIGURES['comparison'], width=15*cm)
if img:
    story.append(Spacer(1, 2*mm))
    story.append(img)
    story.append(Paragraph('Figure 2: Model comparison across validation and cross-validation sets.', style_caption))

# ==================== SECTION 4: Key Features ====================
story.append(Paragraph('4. Key Features', style_h1))
story.append(hr())

story.append(Paragraph(
    'SHAP analysis reveals the Top-20 features driving match outcome predictions:',
    style_body
))

img = add_figure(FIGURES['features'], width=15*cm)
if img:
    story.append(Spacer(1, 2*mm))
    story.append(img)
    story.append(Paragraph('Figure 3: SHAP Top-20 feature importance.', style_caption))

story.append(Paragraph('Top 5 Most Important Factors', style_h2))

top5_data = [
    ['Rank', 'Feature', 'Importance', 'Interpretation'],
    ['1', 'FIFA Rank Difference', '21.6%', 'Direct team strength disparity measure'],
    ['2', 'FIFA Points Difference', '21.4%', 'Continuous complement to rank diff'],
    ['3', 'Away Total Matches', '3.95%', 'Away team experience'],
    ['4', 'Home Total Matches', '3.64%', 'Home team experience'],
    ['5', 'Home FIFA Rank', '3.58%', 'Home team absolute strength'],
]
story.append(make_table(top5_data, col_widths=[1.5*cm, 4.5*cm, 2.5*cm, 5.5*cm]))

story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    '<b>Key finding:</b> FIFA rank difference and points difference together account for '
    '43% of total importance, far exceeding all other features. The single most informative '
    'signal for predicting match outcomes is the gap in team quality. Other influential '
    'factors include neutral venue (3.5%), Elo ratings and Elo difference (5.1% combined), '
    'and recent net goals (2.3%).',
    style_body
))
story.append(Paragraph(
    '<i>Consistent with football intuition: strength disparity &gt; recent form &gt; historical pedigree &gt; venue factors.</i>',
    style_body
))

# ==================== SECTION 5: Prediction Results ====================
story.append(Paragraph('5. Prediction Results', style_h1))
story.append(hr())

story.append(Paragraph('5.1 Champion Probability', style_h2))

img = add_figure(FIGURES['champion'])
if img:
    story.append(img)
    story.append(Paragraph('Figure 4: Top 8 champion probability bar chart.', style_caption))

champ_data = [
    ['Rank', 'Team', 'Confederation', 'Champion Prob.', 'Semifinal Prob.', 'Elo'],
    ['1', 'France', 'UEFA', '28.19%', '72.22%', '1903'],
    ['2', 'Argentina', 'CONMEBOL', '20.76%', '72.60%', '1908'],
    ['3', 'Spain', 'UEFA', '18.24%', '60.92%', '1931'],
    ['4', 'England', 'UEFA', '14.07%', '66.29%', '1873'],
    ['5', 'Belgium', 'UEFA', '6.68%', '39.08%', '1776'],
    ['6', 'Morocco', 'CAF', '6.86%', '27.78%', '1918'],
    ['7', 'Norway', 'UEFA', '2.69%', '33.71%', '1806'],
    ['8', 'Switzerland', 'UEFA', '2.51%', '27.40%', '1752'],
]
story.append(make_table(champ_data, col_widths=[1.2*cm, 2.5*cm, 2.5*cm, 2.8*cm, 2.8*cm, 1.5*cm]))

story.append(Paragraph('5.2 Semifinal Probability', style_h2))

img = add_figure(FIGURES['semifinal'])
if img:
    story.append(img)
    story.append(Paragraph('Figure 5: Semifinal probability distribution.', style_caption))

story.append(Paragraph('5.3 By Confederation', style_h2))

conf_data = [
    ['Confederation', 'Teams', 'Total Prob.', 'Represented By'],
    ['UEFA (Europe)', '5', '69.87%', 'France, Spain, England, Belgium, Norway'],
    ['CONMEBOL (S. America)', '1', '20.76%', 'Argentina'],
    ['CAF (Africa)', '1', '6.86%', 'Morocco'],
    ['CONCACAF (N. America)', '0', '—', '—'],
    ['AFC (Asia)', '0', '—', '—'],
    ['OFC (Oceania)', '0', '—', '—'],
]
story.append(make_table(conf_data, col_widths=[4*cm, 2*cm, 2.5*cm, 5.5*cm]))

img = add_figure(FIGURES['confederation'])
if img:
    story.append(Spacer(1, 2*mm))
    story.append(img)
    story.append(Paragraph('Figure 6: Champion probability by confederation.', style_caption))

story.append(Paragraph('5.4 Knockout Bracket Prediction', style_h2))

img = add_figure(FIGURES['bracket'], width=17*cm)
if img:
    story.append(img)
    story.append(Paragraph('Figure 7: Complete knockout stage prediction bracket.', style_caption))

# QF table
qf_data = [
    ['Matchup', 'Home Win', 'Draw', 'Away Win', 'Forecast'],
    ['France vs Morocco', '61.3%', '22.4%', '16.3%', 'France dominant'],
    ['Norway vs England', '22.5%', '21.8%', '55.7%', 'England advantage'],
    ['Spain vs Belgium', '45.5%', '29.8%', '24.7%', 'Spain slight favorite'],
    ['Argentina vs Switzerland', '61.8%', '20.3%', '17.8%', 'Argentina clear'],
]
story.append(make_table(qf_data, col_widths=[3.5*cm, 2.5*cm, 2*cm, 2.5*cm, 3.5*cm]))

story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    'Most likely semifinal combinations: Argentina vs Spain (44.3%) and England vs France (48.4%).<br/>'
    'Most likely final four: Argentina, England, France, Spain (21.11%).<br/>'
    '<b>Predicted final: France vs Argentina</b> — the European powerhouse against the reigning '
    'South American champion.',
    style_body
))

# ==================== SECTION 6: Robustness ====================
story.append(Paragraph('6. Robustness Analysis', style_h1))
story.append(hr())

story.append(Paragraph('6.1 Different Random Seeds', style_h2))
seed_data = [
    ['Seed', 'Accuracy', 'LogLoss', 'AUC', 'Trees Used'],
    ['42', '0.6264', '0.8409', '0.7610', '339'],
    ['123', '0.6221', '0.8407', '0.7624', '316'],
    ['2024', '0.6298', '0.8387', '0.7608', '352'],
]
story.append(make_table(seed_data, col_widths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
story.append(Paragraph(
    'AUC standard deviation across seeds is only 0.0008; LogLoss SD is 0.0012, '
    'indicating highly stable training.',
    style_body
))

story.append(Paragraph('6.2 Cross-Validation Stability', style_h2))
story.append(Paragraph(
    'Across 5 CV folds, AUC ranges from 0.699 to 0.796 (SD = 0.039), LogLoss from '
    '0.774 to 0.945 (SD = 0.064). The validation set performance closely tracks CV '
    'averages, confirming no severe overfitting.',
    style_body
))

story.append(Paragraph('6.3 Model Limitations', style_h2))
limitations = [
    '<b>Limited data scope:</b> Training set includes 4,622 matches (80% time-series split) '
    'primarily from the 2021–2026 cycle.',
    '<b>External variables excluded:</b> Injuries, suspensions, weather, and other '
    'short-term factors are not incorporated.',
    '<b>Knockout psychology:</b> Penalty shootout pressure and psychological dynamics '
    'in elimination matches are not explicitly modeled.',
    '<b>Tournament specificity:</b> The model is trained on general international matches; '
    'unique aspects of World Cup knockout rounds may not be fully captured.',
]
for lim in limitations:
    story.append(Paragraph(f'&bull; {lim}', style_bullet))

# ==================== SECTION 7: Discussion ====================
story.append(Paragraph('7. Discussion &amp; Outlook', style_h1))
story.append(hr())

story.append(Paragraph('7.1 Champion Prediction Analysis', style_h2))
story.append(Paragraph(
    'France\'s champion probability exceeds one-quarter (28.19%), making them the clear '
    'favorite. Key supporting factors include:',
    style_body
))
factors = [
    '<b>Historical pedigree:</b> Runners-up in the previous World Cup, three finals '
    'in the last four tournaments.',
    '<b>Data profile:</b> Elo rating 1903, 100% win rate in last 10 matches, '
    '2.8 goals per game.',
    '<b>Favorable draw:</b> Quarterfinal against Morocco is the most favorable '
    'matchup among top contenders.',
]
for f in factors:
    story.append(Paragraph(f'&bull; {f}', style_bullet))

story.append(Paragraph('7.2 Uncertainty', style_h2))
story.append(Paragraph(
    'The quarterfinals have yet to be played, and single-elimination matches carry '
    'inherent randomness. The Spain vs Belgium quarterfinal has a 29.8% draw probability — '
    'if extra time or penalties are required, the outcome becomes even harder to predict. '
    'All probabilities reflect statistical inference based on historical data and do not '
    'guarantee actual match results.',
    style_body
))

story.append(Paragraph('7.3 Model Utility', style_h2))
story.append(Paragraph(
    'This study demonstrates a complete data-to-prediction pipeline. Feature engineering '
    'automatically extracts 72 predictive features from raw match data; gradient boosting '
    'achieves CV AUC of 0.749; and Monte Carlo simulation generates interpretable '
    'probability outcomes using the actual tournament schedule. The methodology is '
    'directly transferable to other sports and tournament prediction scenarios.',
    style_body
))

img = add_figure(FIGURES['elo'])
if img:
    story.append(img)
    story.append(Paragraph('Figure 8: Top 12 team Elo ratings with FIFA rank and recent form.', style_caption))

# ==================== SECTION 8: Appendix ====================
story.append(Paragraph('8. Appendix', style_h1))
story.append(hr())

story.append(Paragraph('A. Complete Feature List (72 dimensions)', style_h2))
feat_cats = [
    ['Category', 'Count', 'Description'],
    ['Elo Ratings', '3', 'elo_home_pre, elo_away_pre, elo_diff'],
    ['FIFA Rankings', '6', 'Rank, points, changes (3/6/12 months)'],
    ['Recent Win Rate', '4', 'Last 5/10 match win rate (home/away)'],
    ['Average Goals', '8', 'Last 5/10 avg goals for/against'],
    ['Net Goals', '4', 'Last 5/10 net goals (home/away)'],
    ['Defensive Stability', '4', 'Last 5/10 goals conceded std dev'],
    ['Conversion Rate', '2', 'Shots/goals ratio (home/away)'],
    ['Match Experience', '2', 'Total historical matches played'],
    ['World Cup History', '6', 'Matches, win rate, goals for/against'],
    ['Confederation', '20', 'Home/away confederation (One-hot)'],
    ['Venue Factors', '3', 'Neutral, same confed, home team'],
    ['Rest Days', '2', 'Days since last match'],
    ['Other', '4', 'K-factor, match importance, etc.'],
]
story.append(make_table(feat_cats, col_widths=[3.5*cm, 2*cm, 8.5*cm]))

story.append(Paragraph('B. Technical Specifications', style_h2))
tech_data = [
    ['Item', 'Specification'],
    ['Python', '3.10+'],
    ['Core Dependencies', 'pandas, numpy, scikit-learn, lightgbm, optuna, matplotlib, seaborn'],
    ['Training Samples', '4,622 (2021–2026 matches, 80% time-series split)'],
    ['Validation Samples', '1,175'],
    ['Feature Dimensions', '72'],
    ['Optuna Trials', '150'],
    ['Monte Carlo Simulations', '10,000'],
    ['Random Seed', '42'],
    ['Training Time', '~15 minutes'],
    ['Simulation Time', '~2 minutes'],
]
story.append(make_table(tech_data, col_widths=[4*cm, 10*cm], header=True))

story.append(Spacer(1, 5*mm))
story.append(hr())
story.append(Paragraph(
    '<i>This report was automatically generated by a LightGBM prediction model + '
    'Monte Carlo simulation pipeline. All probabilities are statistical inferences '
    'based on 2021–2026 historical match data. Single-elimination matches carry '
    'inherent randomness; actual results may deviate from predictions.</i>',
    style_footer
))

# ============================================================
# Build
# ============================================================
doc.build(story)
print(f"PDF generated: {PDF_PATH}")
