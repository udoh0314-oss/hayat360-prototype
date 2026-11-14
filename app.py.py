import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import time
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING (ADPHC & INCEPTION42 ALIGNED)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hayat360 | Family Health Passport & Measurable Care Loop",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(90deg, #0f766e 0%, #115e59 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 18px;
    }
    .clean-card {
        background-color: #f8fafc;
        border-left: 4px solid #0f766e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .bilingual-card {
        background-color: #f0fdfa;
        border: 1px solid #99f6e4;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .trace-step {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        padding: 12px 14px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .flag-box {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        padding: 12px 14px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .passport-box {
        background-color: #5c4d44;
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .pill-working {
        background-color: #166534;
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .pill-simulated {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. SYNTHETIC COHORT & 4 ERROR-EXPOSING BENCHMARK CASES (VERIFIED INDEXING)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_cohort_and_model():
    np.random.seed(42)
    n = 1200
    
    cohort_track = np.random.choice(
        ['IFHAS Adult Screening (18-75Y)', 'Well-Child Visits (1M-6Y)'],
        size=n, p=[0.78, 0.22]
    )
    nationalities = np.where(
        cohort_track == 'IFHAS Adult Screening (18-75Y)',
        np.random.choice(['Emirati', 'Arab_Expat', 'South_Asian'], size=n, p=[0.75, 0.15, 0.10]),
        np.random.choice(['Emirati', 'South_Asian', 'Arab_Expat', 'Western_Expat', 'Filipino'], size=n, p=[0.30, 0.30, 0.18, 0.10, 0.12])
    )
    regions = np.random.choice(
        ['Abu Dhabi Region', 'Al Ain Region', 'Al Dhafra Region'],
        size=n, p=[0.55, 0.30, 0.15]
    )
    age = np.where(
        cohort_track == 'Well-Child Visits (1M-6Y)',
        np.round(np.random.uniform(0.2, 6.0, size=n), 1),
        np.random.randint(18, 76, size=n).astype(float)
    )
    weight_kg = np.where(
        cohort_track == 'Well-Child Visits (1M-6Y)',
        np.round(8.0 + age * 2.3 + np.random.normal(0, 1.2, size=n), 1),
        np.round(np.random.normal(79.0, 14.0, size=n), 1)
    )
    
    outdoor_heat_exposure = np.where(
        (cohort_track == 'IFHAS Adult Screening (18-75Y)') & (regions == 'Al Dhafra Region'),
        np.random.choice(['High (Outdoor / Field Work)', 'Standard (Indoor)'], size=n, p=[0.65, 0.35]),
        np.where(cohort_track == 'IFHAS Adult Screening (18-75Y)', np.random.choice(['High (Outdoor / Field Work)', 'Standard (Indoor)'], size=n, p=[0.18, 0.82]), 'Standard (Indoor)')
    )
    
    egp_enrolled = np.where(nationalities == 'Emirati', np.random.binomial(1, 0.82, n), np.random.binomial(1, 0.20, n))
    prs = np.where(egp_enrolled == 1, np.round(np.random.beta(2.2, 4.8, size=n), 3), np.nan)
    pgx_status = np.where(
        egp_enrolled == 1,
        np.random.choice(['SLCO1B1*5 (Statin Sensitivity)', 'Normal Metabolizer'], size=n, p=[0.35, 0.65]),
        'Not Sequenced (Standard Protocol)'
    )
    family_hx = np.random.binomial(1, 0.35, size=n)
    
    base_hba1c = np.where(cohort_track == 'Well-Child Visits (1M-6Y)', 4.9, np.round(np.random.normal(5.3, 0.4, size=n), 2))
    curr_hba1c = np.where(
        cohort_track == 'Well-Child Visits (1M-6Y)',
        np.round(base_hba1c + np.random.normal(0.05, 0.1, size=n), 2),
        np.round(np.clip(base_hba1c + np.random.normal(0.25, 0.32, size=n), 4.6, 9.5), 2)
    )
    delta_hba1c = np.round(curr_hba1c - base_hba1c, 2)
    
    base_bmi = np.where(cohort_track == 'Well-Child Visits (1M-6Y)', 16.2, np.round(np.random.normal(26.5, 4.0, size=n), 1))
    curr_bmi = np.where(
        cohort_track == 'Well-Child Visits (1M-6Y)',
        np.round(base_bmi + np.random.normal(0.3, 0.8, size=n), 1),
        np.round(np.clip(base_bmi + np.random.normal(0.7, 1.2, size=n), 18.5, 42.0), 1)
    )
    delta_bmi = np.round(curr_bmi - base_bmi, 1)
    
    sleep_hours = np.where(cohort_track == 'Well-Child Visits (1M-6Y)', np.round(np.random.normal(10.5, 1.0, size=n), 1), np.round(np.random.normal(6.5, 1.1, size=n), 1))
    months_overdue = np.random.randint(1, 28, size=n)
    
    loop_states = np.random.choice(
        ['02 - Risk Identified (Awaiting Approval)', '03 - Referral Initiated', '04 - Care Completed', '05 - Outcome & Treatment Tracked', 'ALERT - Lost to Follow-Up'],
        size=n, p=[0.22, 0.25, 0.22, 0.19, 0.12]
    )
    days_to_assessment = np.random.randint(2, 24, size=n)
    
    contact_phones = np.random.choice(['+971-50-442-8190', '+971-56-810-3321', ''], size=n, p=[0.48, 0.44, 0.08])
    fasting_status = np.random.choice(['Confirmed Fasting (10h)', 'NPO 10 hrs (nil by mouth since 10 PM)', 'Had tea with milk 2h ago (Unclear)'], size=n, p=[0.70, 0.20, 0.10])
    
    df = pd.DataFrame({
        'patient_id': [f'ADH-{202600+i}' for i in range(n)],
        'family_id': [f'FAM-{1000 + (i // 2)}' for i in range(n)],
        'name': [f'Resident_{i}' for i in range(n)],
        'name_ar': [f'مقيم_{i}' for i in range(n)],
        'test_case_type': ['Synthetic Cohort'] * n,
        'cohort_track': cohort_track,
        'nationality': nationalities,
        'region': regions,
        'age': age,
        'weight_kg': weight_kg,
        'outdoor_heat_exposure': outdoor_heat_exposure,
        'contact_phone': contact_phones,
        'fasting_status': fasting_status,
        'insurance': np.where(nationalities == 'Emirati', 'Thiqa (AED 0 Co-Pay)', 'Daman Enhanced (AED 0 Preventive)'),
        'egp_enrolled': egp_enrolled,
        'prs': prs,
        'pgx_status': pgx_status,
        'family_hx': family_hx,
        'base_hba1c': base_hba1c, 'curr_hba1c': curr_hba1c, 'delta_hba1c': delta_hba1c,
        'base_bmi': base_bmi, 'curr_bmi': curr_bmi, 'delta_bmi': delta_bmi,
        'sleep_hours': sleep_hours,
        'months_overdue': months_overdue,
        'genomic_consent': [True] * n,
        'clinician_approved': np.where(np.isin(loop_states, ['03 - Referral Initiated', '04 - Care Completed', '05 - Outcome & Treatment Tracked']), True, False),
        'status': loop_states,
        'days_to_assessment': days_to_assessment,
        'adherence_days': np.random.randint(0, 30, size=n)
    })

    # 5 EXPLICITLY INDEXED BENCHMARK CASES (ZERO INDEXING ERRORS)
    df.loc[0] = [
        'ADH-202600', 'FAM-1000', 'Tariq Al-Mansoori [Case 1: Complete Referral]', 'طارق المنصوري', '1. Complete referral',
        'IFHAS Adult Screening (18-75Y)', 'Emirati', 'Abu Dhabi Region', 29.0, 86.0,
        'Standard (Indoor)', '+971-50-442-8190', 'Confirmed Fasting (10h)',
        'Thiqa (AED 0 Co-Pay)', 1, 0.84, 'SLCO1B1*5 (Statin Sensitivity)', 1,
        5.20, 5.85, 0.65, 25.8, 28.4, 2.6, 5.4, 19,
        True, False, '02 - Risk Identified (Awaiting Approval)', 6, 14
    ]
    df.loc[1] = [
        'ADH-202601', 'FAM-1001', 'Salma Al-Mazrouei [Case 2: Missing Field]', 'سلمى المزروعي', '2. Required field missing',
        'IFHAS Adult Screening (18-75Y)', 'Emirati', 'Al Dhafra Region', 46.0, 74.0,
        'High (Outdoor / Field Work)', '', 'Confirmed Fasting (10h)',
        'Thiqa (AED 0 Co-Pay)', 1, 0.79, 'Normal Metabolizer', 1,
        5.40, 6.25, 0.85, 27.0, 29.8, 2.8, 5.6, 26,
        True, False, 'ALERT - Lost to Follow-Up', 29, 0
    ]
    df.loc[2] = [
        'ADH-202602', 'FAM-1002', 'Omar Al-Ketbi [Case 3: New Wording "NPO 10 hrs"]', 'عمر الكتبي', '3. Same info, new wording',
        'IFHAS Adult Screening (18-75Y)', 'Emirati', 'Al Ain Region', 38.0, 84.0,
        'High (Outdoor / Field Work)', '+971-56-810-3321', 'NPO 10 hrs (nil by mouth since 10 PM)',
        'Thiqa (AED 0 Co-Pay)', 1, 0.81, 'SLCO1B1*5 (Statin Sensitivity)', 1,
        5.30, 5.95, 0.65, 26.5, 28.9, 2.4, 5.8, 18,
        True, False, '02 - Risk Identified (Awaiting Approval)', 7, 5
    ]
    df.loc[3] = [
        'ADH-202603', 'FAM-1003', 'Priya Nair [Case 4: Unclear Fasting Info]', 'بريا ناير', '4. Unclear information',
        'IFHAS Adult Screening (18-75Y)', 'South_Asian', 'Abu Dhabi Region', 34.0, 68.5,
        'Standard (Indoor)', '+971-55-902-1184', 'Had tea with milk 2h ago (Unclear)',
        'Daman Enhanced (AED 0 Preventive)', 0, np.nan, 'Not Sequenced (Standard Protocol)', 1,
        5.35, 5.90, 0.55, 24.5, 26.8, 2.3, 5.9, 21,
        True, False, '02 - Risk Identified (Awaiting Approval)', 9, 7
    ]
    df.loc[4] = [
        'ADH-202604', 'FAM-1000', 'Zayed Al-Mansoori (Well-Child Visit #11)', 'زايد المنصوري', '1. Complete referral',
