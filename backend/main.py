"""
main.py
-------
FastAPI backend for Global Insure-AI Pro.

Exposes:
  1. POST /api/predict: Computes ML + heuristic insurance premiums.
  2. POST /api/report/pdf: Generates downloadable PDF reports.
  3. POST /api/auth/send-otp: Generates and emails a 6-digit OTP code.
  4. POST /api/auth/verify-otp: Verifies OTP codes and issues access tokens.

Run:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
import joblib
import os
import io
import random
import string
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# Load MAIL_* variables from a local .env file if one exists (see .env.example)
load_dotenv()

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(
    title="Global Insure-AI Pro API",
    description="Hybrid ML + heuristic insurance premium prediction engine with OTP Auth",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, restrict this to your actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FALLBACK_COEFFICIENTS = {
    "intercept": -64741.10712911549,
    "age": 192.026528092248,
    "bmi": 3090.5063104863657,
    "smoker": 6132.242040805674,
    "age_smoker": 278.13227883918745,
}

model_bundle = None  # populated on startup


@app.on_event("startup")
def load_model():
    """Load the trained model into memory once, at server startup."""
    global model_bundle
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
        print(f"[startup] Loaded model from {MODEL_PATH}")
    else:
        model_bundle = None
        print(
            f"[startup] WARNING: '{MODEL_PATH}' not found. "
            "Falling back to hardcoded coefficients. Run check.py to train a real model."
        )


# --------------------------------------------------------------------------
# OTP Email Configuration & In-Memory Store
# --------------------------------------------------------------------------

# Temporary in-memory OTP storage: { "user@example.com": {"code": "123456", "expires_at": datetime} }
# NOTE: this resets whenever the server restarts, and isn't shared across multiple
# server processes/workers. Fine for development; swap for Redis or a DB table in production.
otp_store: dict[str, dict] = {}

OTP_EXPIRY_MINUTES = 5

MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_FROM = os.getenv("MAIL_FROM", MAIL_USERNAME)

# True once real credentials have been supplied via environment variables / .env.
# Until then we deliberately skip attempting SMTP (it would just time out slowly)
# and instead print the OTP to the server console so local development works.
MAIL_IS_CONFIGURED = bool(MAIL_USERNAME) and bool(MAIL_PASSWORD)

mail_config = None
if MAIL_IS_CONFIGURED:
    mail_config = ConnectionConfig(
        MAIL_USERNAME=MAIL_USERNAME,
        MAIL_PASSWORD=MAIL_PASSWORD,
        MAIL_FROM=MAIL_FROM,
        MAIL_PORT=587,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True,
    )
else:
    print(
        "[startup] WARNING: MAIL_USERNAME / MAIL_PASSWORD are not set. "
        "Running in DEV MODE — OTP codes will be printed to this console "
        "instead of being emailed. See .env.example to configure real delivery."
    )


class EmailReq(BaseModel):
    email: EmailStr


class VerifyOTPReq(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


# --------------------------------------------------------------------------
# Request / Response schemas (prediction)
# --------------------------------------------------------------------------

class UserMetrics(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Applicant's full name")
    country: str = Field(..., description="Country of residence, e.g. 'India'")
    age: int = Field(..., ge=18, le=100)
    weight_kg: float = Field(..., gt=0, description="Weight in kilograms")
    height_cm: float = Field(..., gt=0, description="Height in centimeters")
    smoker: bool = Field(..., description="True if the applicant smokes")
    workout_days: int = Field(..., ge=0, le=7, description="Workout days per week")
    diabetes: bool = False
    heart_issue: bool = False
    lung_issue: bool = False
    recent_surgery: bool = False


class PredictionResponse(BaseModel):
    applicant_name: str
    bmi: float
    raw_ml_estimate: float
    risk_multiplier: float
    geo_multiplier: float
    risk_tier: str
    price_usd: float
    price_inr: float
    ai_advice: str
    debug: Optional[dict] = None


# --------------------------------------------------------------------------
# Business logic constants
# --------------------------------------------------------------------------

GEO_MULTIPLIERS = {
    "India": 0.08,
    "USA": 1.0,
    "United Kingdom": 0.65,
    "Canada": 0.70,
    "Australia": 0.75,
    "Germany": 0.68,
}

INR_RATE = 83.0
PREMIUM_FLOOR_USD = 500.0
YOUNG_AGE_THRESHOLD = 30
YOUNG_AGE_DISCOUNT = 0.65

ACTIVE_WORKOUT_THRESHOLD = 5
ACTIVE_DISCOUNT = 0.20
SEDENTARY_PENALTY = 0.10

DIABETES_PENALTY = 0.35
HEART_PENALTY = 1.40
LUNG_PENALTY = 0.50
HIGH_BMI_THRESHOLD = 28
HIGH_BMI_PENALTY = 0.25

HEALTHY_TIER_THRESHOLD = 0.9
CRITICAL_TIER_THRESHOLD = 1.8


# --------------------------------------------------------------------------
# Core calculation helpers
# --------------------------------------------------------------------------

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def run_ml_model(age: int, bmi: float, smoker: bool) -> float:
    smoker_flag = 1 if smoker else 0
    age_smoker = age * smoker_flag

    if model_bundle is not None:
        model = model_bundle["model"]
        features = model_bundle["features"]
        feature_values = {
            "age": age,
            "bmi": bmi,
            "smoker": smoker_flag,
            "age_smoker": age_smoker,
        }
        X = [[feature_values[f] for f in features]]
        raw = float(model.predict(X)[0])
    else:
        c = FALLBACK_COEFFICIENTS
        raw = (
            c["intercept"]
            + c["age"] * age
            + c["bmi"] * bmi
            + c["smoker"] * smoker_flag
            + c["age_smoker"] * age_smoker
        )

    return raw


def apply_heuristics(raw: float, age: int) -> float:
    adjusted = raw
    if age < YOUNG_AGE_THRESHOLD:
        adjusted *= YOUNG_AGE_DISCOUNT
    return max(adjusted, PREMIUM_FLOOR_USD)


def calculate_risk_multiplier(payload: UserMetrics, bmi: float) -> float:
    risk = 1.0

    if payload.workout_days >= ACTIVE_WORKOUT_THRESHOLD:
        risk -= ACTIVE_DISCOUNT
    elif payload.workout_days == 0:
        risk += SEDENTARY_PENALTY

    if payload.diabetes:
        risk += DIABETES_PENALTY
    if payload.heart_issue:
        risk += HEART_PENALTY
    if payload.lung_issue:
        risk += LUNG_PENALTY
    if bmi > HIGH_BMI_THRESHOLD:
        risk += HIGH_BMI_PENALTY

    return risk


def determine_risk_tier(risk_multiplier: float) -> str:
    if risk_multiplier < HEALTHY_TIER_THRESHOLD:
        return "Healthy+"
    if risk_multiplier > CRITICAL_TIER_THRESHOLD:
        return "Critical"
    return "Standard"


def generate_ai_advice(payload: UserMetrics) -> str:
    first_name = payload.name.strip().split(" ")[0] if payload.name.strip() else "there"
    advice_parts = [f"Hi {first_name}!"]
    if payload.smoker:
        advice_parts.append("Quitting smoking could save you up to 45% in premiums.")
    if payload.workout_days < 4:
        advice_parts.append(
            "Working out 5+ days/week would unlock an 'Active Lifestyle' discount."
        )
    if payload.heart_issue:
        advice_parts.append(
            "High risk detected due to cardiac history; regular screenings recommended."
        )
    if len(advice_parts) == 1:
        advice_parts.append(
            "Your current profile looks healthy — keep up the good habits!"
        )
    return " ".join(advice_parts)


def compute_prediction(payload: UserMetrics) -> dict:
    if payload.country not in GEO_MULTIPLIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported country '{payload.country}'. "
            f"Supported countries: {list(GEO_MULTIPLIERS.keys())}",
        )

    bmi = calculate_bmi(payload.weight_kg, payload.height_cm)
    raw = run_ml_model(payload.age, bmi, payload.smoker)
    base = apply_heuristics(raw, payload.age)
    risk_multiplier = calculate_risk_multiplier(payload, bmi)
    geo_multiplier = GEO_MULTIPLIERS[payload.country]
    price_usd = base * risk_multiplier * geo_multiplier
    price_inr = price_usd * INR_RATE
    risk_tier = determine_risk_tier(risk_multiplier)
    ai_advice = generate_ai_advice(payload)

    return {
        "applicant_name": payload.name.strip(),
        "bmi": round(bmi, 1),
        "raw_ml_estimate": round(raw, 2),
        "risk_multiplier": round(risk_multiplier, 2),
        "geo_multiplier": geo_multiplier,
        "risk_tier": risk_tier,
        "price_usd": round(price_usd, 2),
        "price_inr": round(price_inr, 2),
        "ai_advice": ai_advice,
        "debug": {
            "age_smoker_interaction": payload.age * (1 if payload.smoker else 0),
            "model_source": "model.pkl" if model_bundle is not None else "fallback_coefficients",
        },
    }


# --------------------------------------------------------------------------
# Routes: General & Prediction
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Global Insure-AI Pro API is running.",
        "mail_configured": MAIL_IS_CONFIGURED,
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: UserMetrics):
    result = compute_prediction(payload)
    return PredictionResponse(**result)


# --------------------------------------------------------------------------
# Routes: Authentication & OTP
# --------------------------------------------------------------------------

@app.post("/api/auth/send-otp")
async def send_otp(data: EmailReq):
    otp_code = "".join(random.choices(string.digits, k=6))
    otp_store[data.email] = {
        "code": otp_code,
        "expires_at": datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES),
    }

    # Always visible in the server console — handy for local testing even
    # once real email delivery is configured.
    print(f"\n[OTP] {data.email} -> {otp_code} (expires in {OTP_EXPIRY_MINUTES} min)\n")

    if not MAIL_IS_CONFIGURED:
        # Dev mode: no SMTP credentials configured yet. Don't pretend an email
        # was sent — tell the caller plainly so the console code above is
        # discoverable, instead of leaving them waiting on an inbox that will
        # never receive anything.
        return {
            "status": "success",
            "message": f"Dev mode: email delivery isn't configured yet. "
                       f"Check the backend console for the OTP sent to {data.email}.",
            "dev_mode": True,
        }

    message = MessageSchema(
        subject="Your Global Insure-AI Verification Code",
        recipients=[data.email],
        body=f"Your authentication code for Global Insure-AI is: {otp_code}\n\n"
             f"This code expires in {OTP_EXPIRY_MINUTES} minutes.",
        subtype=MessageType.plain,
    )

    try:
        fm = FastMail(mail_config)
        await fm.send_message(message)
    except Exception as e:
        # Surface the real failure instead of silently claiming success —
        # this is exactly what was hiding the root cause before.
        print(f"[mail] FAILED to send OTP email: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Could not send the email. Check your SMTP credentials in .env. ({e})",
        )

    return {"status": "success", "message": f"OTP successfully sent to {data.email}", "dev_mode": False}


@app.post("/api/auth/verify-otp")
async def verify_otp(data: VerifyOTPReq):
    entry = otp_store.get(data.email)

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No verification code requested for this email address.",
        )

    if datetime.utcnow() > entry["expires_at"]:
        del otp_store[data.email]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This code has expired. Please request a new one.",
        )

    if entry["code"] != data.otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That code doesn't look right. Please double-check and try again.",
        )

    del otp_store[data.email]

    return {
        "status": "success",
        "message": "Verification successful",
        "token": f"session-token-{random.randint(100000, 999999)}",
    }


# --------------------------------------------------------------------------
# PDF report generation
# --------------------------------------------------------------------------

def build_pdf_report(payload: UserMetrics, result: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        title="Global Insure-AI Pro - Risk Report",
    )

    styles = getSampleStyleSheet()
    accent = colors.HexColor("#1D9E75")
    accent_dark = colors.HexColor("#0F6E56")
    text_secondary = colors.HexColor("#5F5E5A")
    bg_secondary = colors.HexColor("#F5F5F4")

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], textColor=accent_dark,
        fontSize=20, spaceAfter=2, alignment=TA_LEFT,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], textColor=text_secondary,
        fontSize=10, spaceAfter=14,
    )
    section_heading = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], textColor=accent_dark,
        fontSize=12, spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=15,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8,
        textColor=text_secondary, alignment=TA_CENTER,
    )

    story = []

    story.append(Paragraph("Global Insure-AI Pro", title_style))
    story.append(Paragraph(
        f"Prepared for <b>{result['applicant_name']}</b> &nbsp;|&nbsp; "
        f"Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDCD8")))

    story.append(Paragraph("Applicant Profile", section_heading))
    profile_rows = [
        ["Name", result["applicant_name"], "Country", payload.country],
        ["Age", f"{payload.age} yrs", "Smoker", "Yes" if payload.smoker else "No"],
        ["Weight", f"{payload.weight_kg} kg", "Height", f"{payload.height_cm} cm"],
        ["BMI", str(result["bmi"]), "Workout Days/Week", str(payload.workout_days)],
    ]
    profile_table = Table(profile_rows, colWidths=[38 * mm, 52 * mm, 38 * mm, 52 * mm])
    profile_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), text_secondary),
        ("TEXTCOLOR", (2, 0), (2, -1), text_secondary),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#EDECE8")),
    ]))
    story.append(profile_table)

    story.append(Paragraph("Medical History Declared", section_heading))
    conditions = []
    if payload.diabetes:
        conditions.append("Diabetes")
    if payload.heart_issue:
        conditions.append("Heart Issue")
    if payload.lung_issue:
        conditions.append("Lung Issue")
    if payload.recent_surgery:
        conditions.append("Surgery in last 12 months")
    conditions_text = ", ".join(conditions) if conditions else "None declared"
    story.append(Paragraph(conditions_text, body_style))

    story.append(Paragraph("Risk &amp; Pricing Breakdown", section_heading))
    breakdown_rows = [
        ["Metric", "Value"],
        ["Raw ML Base Estimate", f"${result['raw_ml_estimate']:,.2f}"],
        ["Risk Multiplier", f"{result['risk_multiplier']:.2f}x"],
        ["Geography Multiplier", f"{result['geo_multiplier']}"],
        ["Risk Tier", result["risk_tier"]],
    ]
    breakdown_table = Table(breakdown_rows, colWidths=[90 * mm, 90 * mm])
    breakdown_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_secondary]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDECE8")),
    ]))
    story.append(breakdown_table)

    story.append(Spacer(1, 12))
    premium_rows = [
        ["Estimated Annual Premium", "USD", "INR"],
        ["", f"${result['price_usd']:,.2f}", f"Rs {result['price_inr']:,.2f}"],
    ]
    premium_table = Table(premium_rows, colWidths=[60 * mm, 60 * mm, 60 * mm])
    premium_table.setStyle(TableStyle([
        ("SPAN", (0, 0), (0, 1)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E1F5EE")),
        ("TEXTCOLOR", (0, 0), (-1, 0), accent_dark),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (1, 1), (-1, 1), 15),
        ("TEXTCOLOR", (1, 1), (-1, 1), accent_dark),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, accent),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFE7D8")),
    ]))
    story.append(premium_table)

    story.append(Paragraph("AI Consultant Notes", section_heading))
    story.append(Paragraph(result["ai_advice"], body_style))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDDCD8")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report is an automated estimate generated by a hybrid machine-learning "
        "and heuristic model for informational purposes only. It does not constitute a "
        "binding insurance quote, medical opinion, or financial advice. Actual premiums "
        "are subject to full underwriting review.",
        disclaimer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


@app.post("/api/report/pdf")
def generate_report_pdf(payload: UserMetrics):
    result = compute_prediction(payload)
    pdf_buffer = build_pdf_report(payload, result)

    safe_name = "".join(
        c for c in result["applicant_name"] if c.isalnum() or c in (" ", "-", "_")
    ).strip().replace(" ", "-") or "Applicant"
    filename = f"Global-Insure-AI-Report-{safe_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
