import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from config import (
    CAREER_ROLES, CAREER_INFO, CAREER_RELEVANT_SKILLS,
    STRONG_ALIGNMENTS, PERSONALITY_TRAITS, SKILLS, INTERESTS
)

# Load artifacts 
model                  = joblib.load("career_model.joblib")
feature_columns        = joblib.load("feature_columns.joblib")
risk_tolerance_mapping = joblib.load("risk_tolerance_mapping.joblib")

# Relevant job roles 
def get_relevant_roles(cluster, education, field):
    edu_roles = CAREER_ROLES.get(cluster, {}).get(education, {})
    seen, roles = set(), []
    for r in edu_roles.get(field, []) + edu_roles.get("Any", []):
        if r not in seen and len(roles) < 8:
            roles.append(r)
            seen.add(r)
    return roles or ["Various professional roles in this field"]

#Decision rationale
def explain_prediction(inp, career):
    explanations = []
    interests = {
        "technology": inp["interest_technology"],
        "business": inp["interest_business"],
        "creative": inp["interest_creative"],
        "health & social": inp["interest_health_social"],
        "research": inp["interest_research_academic"]
    }
    top_k, top_v = max(interests.items(), key=lambda x: x[1])
    if top_v >= 50:
        strength = "strong " if top_v >= 70 else ""
        explanations.append(
            f"Your {strength}interest in **{top_k}** ({top_v}/100) fits this career"
        )
        
    skills = {
        "Technical": inp["technical_skill"],
        "Data Reasoning": inp["data_reasoning_skill"],
        "Communication": inp["communication_skill"],
        "Problem Solving": inp["problem_solving_skill"],
        "Leadership": inp["leadership_skill"],
        "Creative Thinking": inp["creative_thinking_skill"]
    }
    relevant = CAREER_RELEVANT_SKILLS.get(career, [])
    top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
    for name, val in top_skills:
        if name in relevant and val >= 3 and len(explanations) < 3:
            level = "excellent" if val >= 4 else "good"
            explanations.append(f"Your {level} {name} skills ({val}/5) are valuable here")
    
    traits = {
        "Openness": inp["openness"],
        "Conscientiousness": inp["conscientiousness"],
        "Extraversion": inp["extraversion"],
        "Agreeableness": inp["agreeableness"]}
    career_traits = {
        "Technology & Engineering": ("Openness", "fits fast-changing tech work"),
        "Healthcare & Life Sciences": ("Agreeableness", "matches people-focused roles"),
        "Business & Management": ("Extraversion", "supports leadership work"),
        "Design & Creative Media": ("Openness", "supports creativity"),
        "Finance & Economics": ("Conscientiousness", "fits analytical work"),
        "Entrepreneurship & Freelance": ("Openness", "fits independent mindset"),
        "Education & Social Impact": ("Agreeableness", "fits mentoring roles")}
    if career in career_traits:
        trait, msg = career_traits[career]
        if traits[trait] >= 4:
            explanations.append(f"Your {msg}")

    ws = inp["preferred_work_style"]
    env = inp["preferred_environment"]
    risk = inp["risk_tolerance"]
    if env == "flexible" and career in ["Design & Creative Media", "Entrepreneurship & Freelance"]:
        explanations.append("You prefer a flexible environment")
    if env == "structured" and career in ["Finance & Economics", "Healthcare & Life Sciences"]:
        explanations.append("You prefer structured work")
    if risk == 2 and career == "Entrepreneurship & Freelance":
        explanations.append("You have high risk tolerance")
    
    fos = inp["field_of_study"]
    exp = inp["experience_years"]
    tech_fields = ["Computer Science", "IT", "Engineering"]
    biz_fields = ["Business", "Economics"]
    if fos in tech_fields and career == "Technology & Engineering":
        explanations.append(f"Your {fos} background supports this career")
    if fos in biz_fields and career in ["Business & Management", "Finance & Economics"]:
        explanations.append(f"Your {fos} background fits this field")
    if exp >= 3:
        explanations.append(f"Your {exp} years of experience is an advantage")


    return explanations or [
        "Your skills and interests fit this field",
        "This career matches your overall profile"]
#_____________________________________________________________________
def star(n): return "⭐" * n
def career_card_html(icon, color, rank_label, name):
    return f"""
    <div style='border:2px solid {color}; border-radius:12px; padding:20px;
                text-align:center; background:linear-gradient(135deg,{color}15 0%,{color}05 100%);'>
        <div style='font-size:2em; margin-bottom:8px;'>{icon}</div>
        <div style='color:{color}; font-weight:600; font-size:0.85em;
                    text-transform:uppercase; letter-spacing:1px;'>{rank_label}</div>
        <div style='font-weight:700; font-size:1.05em; margin:8px 0; color:#FFF;'>{name}</div>
    </div>"""


def header_card_html(icon, color, rank_label, name):
    return f"""
    <div style='text-align:center; padding:30px;
                background:linear-gradient(135deg,{color}22 0%,{color}11 100%);
                border-radius:15px; margin-bottom:20px;'>
        <h1 style='margin:0; font-size:3em;'>{icon}</h1>
        <h2 style='margin:10px 0; color:{color};'>{rank_label}</h2>
        <h1 style='margin:0; color:#FFF; font-weight:bold;'>{name}</h1>
    </div>"""

#_______________________________________________________________________
# Page config 
st.title("Personality and Competency-Aligned Career Recommendation")
st.markdown("""
    <style>
        .block-container {
            max-width: 1400px !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }
        div[data-testid="stSlider"] {
            padding-right: 1.5rem;
        }
        div[data-testid="stSlider"] label p {
            font-size: 0.95rem !important;
            white-space: nowrap;
        }
        section[data-testid="stSidebar"] {
            min-width: 0px;
        }
        div[data-testid="column"] {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .stSlider {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("### Answer the questions below to get your personalized career path suggestion")
st.markdown("---")

#Form
with st.form("profile_form"):
    st.subheader(" Personal Information")
    c1, c2 = st.columns(2)
    user_id   = c1.text_input("User ID", placeholder="Enter User ID")
    user_name = c2.text_input("Name",    placeholder="Enter Name")
    st.markdown("---")

    # Basic Info
    st.subheader(" Basic Information")
    b1, b2, b3, b4 = st.columns(4)
    age        = b1.number_input("Age", 18, 65, 24)
    education  = b2.selectbox("Education Level", ["High School", "Undergraduate", "Graduate"])
    field      = b3.selectbox("Field of Study",  ["Computer Science", "IT", "Engineering", "Business",
                                                   "Economics", "Psychology", "Design", "Biology", "Other"])
    exp_input  = b4.text_input("Years of Experience", value="2")
    st.markdown("---")

    # Personality
    st.subheader(" Personality Traits")
    st.caption("Rate yourself on a scale of 1 (Low) to 5 (High)")
    p_vals = {}
    for col, (trait, help_txt) in zip(st.columns(5,gap="small"), PERSONALITY_TRAITS.items()):
        p_vals[trait.lower()] = col.slider(trait, 1, 5, 3, help=help_txt)
    st.markdown("---")

    # Skills
    st.subheader(" Skills Assessment")
    st.caption("Rate your proficiency from 0 (None) to 5 (Expert)")
    s_vals = {}
    for col, skill in zip(st.columns(6, gap="small"), SKILLS):
        s_vals[skill] = col.slider(skill, 0, 5, 3)
    st.markdown("---")

    # Interests
    st.subheader(" Interest Areas")
    st.caption("Rate your interest level from 0 (Not Interested) to 100 (Highly Interested)")
    i_vals = {}
    for col, interest in zip(st.columns(5), INTERESTS):
        i_vals[interest] = col.slider(interest, 0, 100, 50)
    st.markdown("---")

    # Work Preferences
    st.subheader(" Work Preferences")
    w1, w2, w3 = st.columns(3)
    work_style = w1.radio("Preferred Work Style", ["individual", "collaborative"])
    env        = w2.radio("Preferred Environment", ["structured", "flexible"])
    risk       = w3.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
    st.markdown("---")

    submitted = st.form_submit_button("Get My Career Recommendation", type="primary", use_container_width=True)


# Results
if submitted:
    try:
        experience = float(str(exp_input).replace("years", "").replace("year", "").strip())
    except:
        experience = 0
    experience = max(0, min(experience, max(0, age - 18)))

    # Validation warnings
    if sum(i_vals.values()) < 50 or sum(s_vals.values()) < 5:
        st.warning(" Your profile shows limited interests/skills. Consider a more detailed assessment.")
    risk_numeric = risk_tolerance_mapping[risk] 
    #Create input dict
    input_dict = {
    "age":                    age,
    "experience_years":       experience,
    "openness":               p_vals["openness"],
    "conscientiousness":      p_vals["conscientiousness"],
    "extraversion":           p_vals["extraversion"],
    "agreeableness":          p_vals["agreeableness"],
    "neuroticism":            p_vals["neuroticism"],
    "technical_skill":        s_vals["Technical"],
    "data_reasoning_skill":   s_vals["Data Reasoning"],
    "communication_skill":    s_vals["Communication"],
    "problem_solving_skill":  s_vals["Problem Solving"],
    "leadership_skill":       s_vals["Leadership"],
    "creative_thinking_skill":s_vals["Creative Thinking"],
    "interest_technology":    i_vals["Technology"],
    "interest_business":      i_vals["Business"],
    "interest_creative":      i_vals["Creative/Arts"],
    "interest_health_social": i_vals["Health & Social"],
    "interest_research_academic": i_vals["Research/Academic"],
    "risk_tolerance":         risk_numeric,
    "education_level":        education,
    "field_of_study":         field if field in ['Computer Science','IT','Engineering','Business',
                                                 'Economics','Psychology','Design','Biology'] else "Other",
    "preferred_work_style":   work_style,
    "preferred_environment":  env,
    }

    # Feature engineering + encode
    df_input = pd.DataFrame([input_dict])
    df_input["work_life_alignment"] = (
        df_input["preferred_environment"].eq("flexible").astype(int) -
        df_input["preferred_environment"].eq("structured").astype(int) +
        df_input["risk_tolerance"].eq(0).astype(int) -
        df_input["risk_tolerance"].eq(2).astype(int)
    )
    categorical_features = ["education_level", "field_of_study", "preferred_work_style", "preferred_environment"]
    df_encoded = pd.get_dummies(df_input, columns=categorical_features, drop_first=True)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    pred         = model.predict(df_aligned)[0]
    probabilities = model.predict_proba(df_aligned)[0]
    careers      = model.classes_
    top3         = np.argsort(probabilities)[-3:][::-1]
    confidence   = probabilities[top3[0]]

    # Ambiguity warning
    interest_vals = list(i_vals.values())
    if confidence < 0.4 or (max(interest_vals) - min(interest_vals)) <= 10:
        st.warning("**Ambiguous Profile Detected**: Showing your top 3 career matches to help you explore options.")

    # ── Top 3 summary cards ────────────────────────────────────────────────
    st.markdown("### 🎯 Your Top Career Matches")
    st.markdown("---")
    rank_labels = ["1st Match", "2nd Match", "3rd Match"]
    for col, idx, label in zip(st.columns(3), top3, rank_labels):
        c = CAREER_INFO.get(careers[idx], {})
        col.markdown(career_card_html(c.get("icon","🎯"), c.get("color","#4A90E2"), label, careers[idx]), unsafe_allow_html=True)

    # Field-career alignment hint
    expected = STRONG_ALIGNMENTS.get(field, [])
    if expected and not any(careers[i] in expected for i in top3):
        st.warning(f"**Unexpected Match**: Based on your **{field}** background, also explore: {', '.join(expected)}")

    # User header
    if user_name or user_id:
        st.markdown("---")
        st.markdown(f"### 👋 Hello, **{user_name or 'there'}**!" + (f"  **ID:** `{user_id}`" if user_id else ""))

    # ── Profile overview ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Profile Overview")
    ov1, ov2, ov3 = st.columns(3)

    with ov1:
        st.markdown("#####  Demographics")
        st.info(f"**Age:** {age}y  \n**Education:** {education}  \n**Field:** {field}  \n**Experience:** {experience}y")
        st.markdown("#####  Work Preferences")
        st.info(f"**Work Style:** {work_style.capitalize()}  \n**Environment:** {env.capitalize()}  \n**Risk:** {risk}")

    with ov2:
        st.markdown("#####  Personality")
        for trait, val in p_vals.items():
            st.write(f"**{trait.capitalize()}:** {star(val)}")
        st.markdown("#####  Skills")
        for skill, val in s_vals.items():
            st.write(f"**{skill}:** {star(val)}")

    with ov3:
        st.markdown("#####  Interests")
        for interest, val in i_vals.items():
            st.write(f"**{interest}:** {val}/100")
            st.progress(val / 100)

    #Full career cards 
    full_rank_labels = ["Primary Recommendation", " Second Recommendation", "Third Recommendation"]
    for rank, idx in enumerate(top3):
        name = careers[idx]
        cd   = CAREER_INFO.get(name, {"icon":"🎯","color":"#4A90E2","description":"","key_skills":[],"salary_range":"","growth_outlook":"","work_style":""})

        st.markdown("---")
        st.markdown(header_card_html(cd["icon"], cd["color"], full_rank_labels[rank], name), unsafe_allow_html=True)

        st.markdown("### 📖 About This Career Path")
        st.info(cd["description"])

        r1, r2 = st.columns([3, 2])
        with r1:
            st.markdown("### 💼 Example Career Roles")
            st.caption(f"Based on your education ({education}) and field ({field})")
            for i, role in enumerate(get_relevant_roles(name, education, field), 1):
                st.markdown(f"{i}. **{role}**")
        
    # Decision rationale
    st.markdown("---")
    st.markdown("### 🔍 Decision Rationale")
    st.caption("Based on your profile, here's why this career suits you:")
    explanations = explain_prediction(input_dict, pred)
    for reason in explanations:
        st.markdown(f"✓ {reason}")

# Download report
    st.markdown("---")
    st.subheader("📄 Save Your Results")
    if user_id or user_name:
        selected_options = "\n".join(f"{k.replace('_',' ').title()}: {v}" for k,v in input_dict.items())
        top3_clusters   = "\n".join(f"{i}. {careers[idx]}" for i,idx in enumerate(top3,1))
        report = (
            f"CAREER REPORT\n"
            f"{'='*35}\n\n"
            f"Name: {user_name or 'N/A'}    ID: {user_id or 'N/A'}\n\n"
            f"SELECTED OPTIONS\n"
            f"{'-'*16}\n"
            f"{selected_options}\n\n"
            f"TOP 3 CAREER CLUSTERS\n"
            f"{'-'*20}\n"
            f"{top3_clusters}\n"
        )

        st.download_button(" Download", report, f"career_report_{user_id or 'anonymous'}.txt", use_container_width=True)
    else:
        st.info("Enter Name or ID to enable download")