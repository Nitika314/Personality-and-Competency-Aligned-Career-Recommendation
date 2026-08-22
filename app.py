import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ollama
import pickle
from config import (
    CAREER_ROLES, CAREER_INFO, CAREER_RELEVANT_SKILLS,
    STRONG_ALIGNMENTS, PERSONALITY_TRAITS, SKILLS, INTERESTS
)

# ══════════════════════════════════════════════════════════
#  STEP 1 — ML models + load RAG knowledge base
# ══════════════════════════════════════════════════════════
model                  = joblib.load("career_model.joblib")
feature_columns        = joblib.load("feature_columns.joblib")
risk_tolerance_mapping = joblib.load("risk_tolerance_mapping.joblib")

@st.cache_resource
def load_knowledge():
    with open("knowledge.pkl", "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]

chunks, doc_embeddings = load_knowledge()

# ══════════════════════════════════════════════════════════
#  STEP 2 — cosine_similarity + retrieve_chunks helpers
# ══════════════════════════════════════════════════════════
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve_chunks(query: str, k: int = 4) -> str:
    resp            = ollama.embeddings(model="nomic-embed-text", prompt=query)
    query_embedding = resp["embedding"]
    sims            = [cosine_similarity(query_embedding, e) for e in doc_embeddings]
    top_k           = np.argsort(sims)[-k:][::-1]
    return "\n---\n".join([chunks[i] for i in top_k])

def build_profile_text(inp: dict) -> str:
    return (
        f"Big Five (1-5): Openness={inp['openness']}, "
        f"Conscientiousness={inp['conscientiousness']}, "
        f"Extraversion={inp['extraversion']}, "
        f"Agreeableness={inp['agreeableness']}, "
        f"Neuroticism={inp['neuroticism']}\n"
        f"Skills (0-5): Technical={inp['technical_skill']}, "
        f"Data={inp['data_reasoning_skill']}, "
        f"Communication={inp['communication_skill']}, "
        f"ProblemSolving={inp['problem_solving_skill']}, "
        f"Leadership={inp['leadership_skill']}, "
        f"Creative={inp['creative_thinking_skill']}\n"
        f"Interests (0-100): Tech={inp['interest_technology']}, "
        f"Business={inp['interest_business']}, "
        f"Creative={inp['interest_creative']}, "
        f"Health={inp['interest_health_social']}, "
        f"Research={inp['interest_research_academic']}\n"
        f"Background: Age={inp['age']}, Education={inp['education_level']}, "
        f"Field={inp['field_of_study']}, Experience={inp['experience_years']}y, "
        f"WorkStyle={inp['preferred_work_style']}, "
        f"Environment={inp['preferred_environment']}, "
        f"Risk={inp['risk_tolerance']}"
    )

# ══════════════════════════════════════════════════════════
#  STEP 3 — explain_prediction with RAG
# ══════════════════════════════════════════════════════════
def explain_prediction(inp: dict, career: str) -> list:
    profile_text = build_profile_text(inp)
    context      = retrieve_chunks(f"{profile_text}\nPredicted career: {career}")

    prompt = f"""You are a career counselor. Using the knowledge base and the 
user's actual scores below, write exactly 4 short bullet points explaining 
why the predicted career cluster suits this specific person.
Reference their actual numbers. No markdown asterisks. 
Start each bullet with a dash (-).

Knowledge Base:
{context}

User Profile:
{profile_text}

Predicted Career: {career}

4 personalised reasons:"""

    try:
        out   = ollama.generate(model="gemma3:1b", prompt=prompt,
                                options={"temperature": 0.2})
        lines = [l.lstrip("-• ").strip() for l in out["response"].split("\n")
                 if l.strip() and l.strip()[0] in ("-", "•")]
        return lines if lines else [out["response"].strip()]
    except Exception:
        return [
            "Your skills and interests align with this career cluster.",
            "Your personality profile matches professionals in this field.",
            "Your education and experience support this direction.",
            "Your work preferences suit this career environment."
        ]

# ══════════════════════════════════════════════════════════
#  STEP 4 — rag_chat with selected career
# ══════════════════════════════════════════════════════════
def rag_chat(question: str, inp: dict, selected_career: str) -> str:
    profile_text = build_profile_text(inp)
    context      = retrieve_chunks(
        f"{profile_text}\nCareer cluster of interest: {selected_career}\nQuestion: {question}"
    )
    prompt = f"""You are an expert career advisor. Answer the career question 
using the knowledge base and the user's profile. Be specific, practical, 
and encouraging. Keep the answer under 150 words.

Knowledge Base:
{context}

User Profile:
{profile_text}

Career Cluster the user is asking about: {selected_career}

Question: {question}

Answer:"""

    try:
        out = ollama.generate(model="gemma3:1b", prompt=prompt,
                              options={"temperature": 0.3})
        return out["response"].strip()
    except Exception:
        return "Ollama is not running. Please start Ollama and try again."

# ══════════════════════════════════════════════════════════
#  EXISTING HELPERS
# ══════════════════════════════════════════════════════════
def get_relevant_roles(cluster, education, field):
    edu_roles = CAREER_ROLES.get(cluster, {}).get(education, {})
    seen, roles = set(), []
    for r in edu_roles.get(field, []) + edu_roles.get("Any", []):
        if r not in seen and len(roles) < 8:
            roles.append(r)
            seen.add(r)
    return roles or ["Various professional roles in this field"]

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

# ══════════════════════════════════════════════════════════
#  UI — PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════
st.title("Personality and Competency-Aligned Career Recommendation")
st.markdown("""
    <style>
        .block-container {
            max-width: 1400px !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }
        div[data-testid="stSlider"] { padding-right: 1.5rem; }
        div[data-testid="stSlider"] label p {
            font-size: 0.95rem !important;
            white-space: nowrap;
        }
        section[data-testid="stSidebar"] { min-width: 0px; }
        div[data-testid="column"] {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .stSlider { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)
st.markdown("### Answer the questions below to get your personalized career path suggestion")
st.markdown("---")

# ══════════════════════════════════════════════════════════
#  FORM
# ══════════════════════════════════════════════════════════
with st.form("profile_form"):
    st.subheader("Personal Information")
    c1, c2    = st.columns(2)
    user_id   = c1.text_input("User ID",  placeholder="Enter User ID")
    user_name = c2.text_input("Name",     placeholder="Enter Name")
    st.markdown("---")

    st.subheader("Basic Information")
    b1, b2, b3, b4 = st.columns(4)
    age       = b1.number_input("Age", 18, 65, 24)
    education = b2.selectbox("Education Level",
                             ["High School", "Undergraduate", "Graduate"])
    field     = b3.selectbox("Field of Study",
                             ["Computer Science","IT","Engineering","Business",
                              "Economics","Psychology","Design","Biology","Other"])
    exp_input = b4.text_input("Years of Experience", value="2")
    st.markdown("---")

    st.subheader("Personality Traits")
    st.caption("Rate yourself on a scale of 1 (Low) to 5 (High)")
    p_vals = {}
    for col, (trait, help_txt) in zip(st.columns(5, gap="small"), PERSONALITY_TRAITS.items()):
        p_vals[trait.lower()] = col.slider(trait, 1, 5, 3, help=help_txt)
    st.markdown("---")

    st.subheader("Skills Assessment")
    st.caption("Rate your proficiency from 0 (None) to 5 (Expert)")
    s_vals = {}
    for col, skill in zip(st.columns(6, gap="small"), SKILLS):
        s_vals[skill] = col.slider(skill, 0, 5, 3)
    st.markdown("---")

    st.subheader("Interest Areas")
    st.caption("Rate your interest level from 0 (Not Interested) to 100 (Highly Interested)")
    i_vals = {}
    for col, interest in zip(st.columns(5), INTERESTS):
        i_vals[interest] = col.slider(interest, 0, 100, 50)
    st.markdown("---")

    st.subheader("Work Preferences")
    w1, w2, w3 = st.columns(3)
    work_style = w1.radio("Preferred Work Style", ["individual", "collaborative"])
    env        = w2.radio("Preferred Environment", ["structured", "flexible"])
    risk       = w3.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
    st.markdown("---")

    submitted = st.form_submit_button(
        "Get My Career Recommendation", type="primary", use_container_width=True
    )

# ══════════════════════════════════════════════════════════
#  PROCESS FORM SUBMISSION
# ══════════════════════════════════════════════════════════
if submitted:
    try:
        experience = float(str(exp_input).replace("years","").replace("year","").strip())
    except Exception:
        experience = 0
    experience = max(0, min(experience, max(0, age - 18)))

    risk_numeric = risk_tolerance_mapping[risk]

    input_dict = {
        "age":                        age,
        "experience_years":           experience,
        "openness":                   p_vals["openness"],
        "conscientiousness":          p_vals["conscientiousness"],
        "extraversion":               p_vals["extraversion"],
        "agreeableness":              p_vals["agreeableness"],
        "neuroticism":                p_vals["neuroticism"],
        "technical_skill":            s_vals["Technical"],
        "data_reasoning_skill":       s_vals["Data Reasoning"],
        "communication_skill":        s_vals["Communication"],
        "problem_solving_skill":      s_vals["Problem Solving"],
        "leadership_skill":           s_vals["Leadership"],
        "creative_thinking_skill":    s_vals["Creative Thinking"],
        "interest_technology":        i_vals["Technology"],
        "interest_business":          i_vals["Business"],
        "interest_creative":          i_vals["Creative/Arts"],
        "interest_health_social":     i_vals["Health & Social"],
        "interest_research_academic": i_vals["Research/Academic"],
        "risk_tolerance":             risk_numeric,
        "education_level":            education,
        "field_of_study":             field if field in [
            "Computer Science","IT","Engineering","Business",
            "Economics","Psychology","Design","Biology"] else "Other",
        "preferred_work_style":       work_style,
        "preferred_environment":      env,
    }

    # ML preprocessing
    df_input = pd.DataFrame([input_dict])
    df_input["work_life_alignment"] = (
        df_input["preferred_environment"].eq("flexible").astype(int) -
        df_input["preferred_environment"].eq("structured").astype(int) +
        df_input["risk_tolerance"].eq(0).astype(int) -
        df_input["risk_tolerance"].eq(2).astype(int)
    )
    categorical_features = ["education_level","field_of_study",
                             "preferred_work_style","preferred_environment"]
    df_encoded = pd.get_dummies(df_input, columns=categorical_features, drop_first=True)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    pred          = model.predict(df_aligned)[0]
    probabilities = model.predict_proba(df_aligned)[0]
    careers       = model.classes_
    top3_indices  = np.argsort(probabilities)[-3:][::-1]
    top3_clusters = [careers[i] for i in top3_indices]
    top3_probs    = [probabilities[i] for i in top3_indices]
    confidence    = top3_probs[0]

    # Generate explanations
    with st.spinner("Generating personalised explanation..."):
        explanations = explain_prediction(input_dict, pred)

    # Store everything in session_state
    st.session_state["show_recommendation"] = True
    st.session_state["user_id"]             = user_id
    st.session_state["user_name"]           = user_name
    st.session_state["input_dict"]          = input_dict
    st.session_state["predicted_career"]    = pred
    st.session_state["top3_clusters"]       = top3_clusters
    st.session_state["top3_probs"]          = top3_probs
    st.session_state["top3_indices"]        = top3_indices
    st.session_state["careers"]             = careers
    st.session_state["explanations"]        = explanations
    st.session_state["confidence"]          = confidence
    st.session_state["field_of_study"]      = field
    st.session_state["education_level"]     = education
    st.session_state["risk"]               = risk
    st.session_state["work_style"]          = work_style
    st.session_state["env"]                = env
    st.session_state["age"]               = age
    st.session_state["p_vals"]             = p_vals
    st.session_state["s_vals"]             = s_vals
    st.session_state["i_vals"]             = i_vals
    st.session_state["discuss_cluster"]    = top3_clusters[0]

    # Build initial chat messages
    initial_messages = []

    # System note (hidden from user) if unexpected match
    expected = STRONG_ALIGNMENTS.get(field, [])
    if expected and not any(careers[i] in expected for i in top3_indices):
        warning_msg = f"Based on your **{field}** background, also explore: {', '.join(expected)}"
        st.session_state["unexpected_match_warning"] = warning_msg
        initial_messages.append({
            "role": "system",
            "content": (
                f"[System note: The user's background in {field} typically aligns with "
                f"{', '.join(expected)}. Suggest these as alternative paths if relevant.]"
            )
        })
    else:
        st.session_state["unexpected_match_warning"] = None

    # Visible assistant intro message
    initial_messages.append({
        "role": "assistant",
        "content": (
            f"Hi{' ' + user_name if user_name else ''}! 👋 Your profile points to "
            f"**{pred}** as your top match. "
            f"I'm currently set to discuss **{top3_clusters[0]}** — "
            "but you can switch clusters using the dropdown above. "
            "Ask me anything about this career path, skills to build, or next steps!"
        )
    })

    st.session_state["chat_messages"] = initial_messages
    st.rerun()

# ══════════════════════════════════════════════════════════
#  RENDER RECOMMENDATION SECTION
# ══════════════════════════════════════════════════════════
if st.session_state.get("show_recommendation", False):

    # Retrieve stored data
    user_id       = st.session_state.get("user_id", "")
    user_name     = st.session_state.get("user_name", "")
    input_dict    = st.session_state["input_dict"]
    pred          = st.session_state["predicted_career"]
    top3_clusters = st.session_state["top3_clusters"]
    top3_probs    = st.session_state["top3_probs"]
    top3_indices  = st.session_state["top3_indices"]
    careers       = st.session_state["careers"]
    explanations  = st.session_state["explanations"]
    confidence    = st.session_state["confidence"]
    field         = st.session_state["field_of_study"]
    education     = st.session_state["education_level"]
    risk          = st.session_state["risk"]
    work_style    = st.session_state["work_style"]
    env           = st.session_state["env"]
    age           = st.session_state["age"]
    p_vals        = st.session_state["p_vals"]
    s_vals        = st.session_state["s_vals"]
    i_vals        = st.session_state["i_vals"]
    warning_msg   = st.session_state.get("unexpected_match_warning")

    # Warnings
    if confidence < 0.4 or (max(i_vals.values()) - min(i_vals.values())) <= 10:
        st.warning("**Ambiguous Profile Detected**: Showing your top 3 career matches.")
    if warning_msg:
        st.warning(warning_msg)

    # Top 3 cards
    st.markdown("### Your Top Career Matches")
    st.markdown("---")
    rank_labels = ["1st Match", "2nd Match", "3rd Match"]
    for col, idx, label in zip(st.columns(3), top3_indices, rank_labels):
        c = CAREER_INFO.get(careers[idx], {})
        col.markdown(
            career_card_html(c.get("icon","🎯"), c.get("color","#4A90E2"), label, careers[idx]),
            unsafe_allow_html=True
        )

    if user_name or user_id:
        st.markdown("---")
        st.markdown(f"### Hello, **{user_name or 'there'}**!")

    # Profile overview
    st.markdown("---")
    st.subheader("Profile Overview")
    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        st.markdown("##### Demographics")
        st.info(
            f"**Age:** {age}y  \n**Education:** {education}  \n"
            f"**Field:** {field}  \n**Experience:** {input_dict['experience_years']}y"
        )
        st.markdown("##### Work Preferences")
        st.info(
            f"**Work Style:** {work_style.capitalize()}  \n"
            f"**Environment:** {env.capitalize()}  \n**Risk:** {risk}"
        )
    with ov2:
        st.markdown("##### Personality")
        for trait, val in p_vals.items():
            st.write(f"**{trait.capitalize()}:** {star(val)}")
        st.markdown("##### Skills")
        for skill, val in s_vals.items():
            st.write(f"**{skill}:** {star(val)}")
    with ov3:
        st.markdown("##### Interests")
        for interest, val in i_vals.items():
            st.write(f"**{interest}:** {val}/100")
            st.progress(val / 100)

    # Per-career detail cards
    full_rank_labels = [
        "Primary Recommendation",
        "Second Recommendation",
        "Third Recommendation"
    ]
    for rank, idx in enumerate(top3_indices):
        name = careers[idx]
        cd   = CAREER_INFO.get(name, {"icon":"🎯","color":"#4A90E2","description":""})
        st.markdown("---")
        st.markdown(
            header_card_html(cd["icon"], cd["color"], full_rank_labels[rank], name),
            unsafe_allow_html=True
        )
        st.markdown("### About This Career Path")
        st.info(cd["description"])
        r1, r2 = st.columns([3, 2])
        with r1:
            st.markdown("### Example Career Roles")
            st.caption(f"Based on your education ({education}) and field ({field})")
            for i, role in enumerate(get_relevant_roles(name, education, field), 1):
                st.markdown(f"{i}. **{role}**")

    # Decision Rationale
    st.markdown("---")
    st.markdown("### Decision Rationale")
    st.caption("AI-generated explanation personalised to your exact scores")
    for reason in explanations:
        st.markdown(f"✓ {reason}")

    # Download report
    st.markdown("---")
    st.subheader("Save Your Results")
    if user_id or user_name:
        selected_options  = "\n".join(
            f"{k.replace('_',' ').title()}: {v}" for k, v in input_dict.items()
        )
        top3_clusters_str = "\n".join(
            f"{i+1}. {name} ({top3_probs[i]*100:.1f}%)"
            for i, name in enumerate(top3_clusters)
        )
        explanation_text = "\n".join(f"- {r}" for r in explanations)

        report = (
            f"CAREER REPORT\n{'='*35}\n\n"
            f"Name: {user_name or 'N/A'}    ID: {user_id or 'N/A'}\n\n"
            f"SELECTED OPTIONS\n{'-'*16}\n{selected_options}\n\n"
            f"TOP 3 CAREER CLUSTERS\n{'-'*20}\n{top3_clusters_str}\n\n"
            f"AI CAREER ADVISOR EXPLANATION\n{'-'*29}\n{explanation_text}\n"
        )
        st.download_button(
            "Download Report", report,
            f"career_report_{user_id or 'anonymous'}.txt",
            use_container_width=True
        )
    else:
        st.info("Enter Name or ID to enable download")

# ══════════════════════════════════════════════════════════
#  CHAT SECTION (only if recommendation exists)
# ══════════════════════════════════════════════════════════
if st.session_state.get("show_recommendation", False):
    st.markdown("---")
    st.subheader("💬 Ask the Career Advisor")

    top3 = st.session_state.get("top3_clusters", [])

    if top3:
        current_discuss = st.session_state.get("discuss_cluster", top3[0])
        discuss_idx     = top3.index(current_discuss) if current_discuss in top3 else 0

        selected_cluster = st.selectbox(
            "Which career cluster would you like to ask about?",
            options=top3,
            index=discuss_idx,
            key="cluster_selector"
        )

        # Reset chat only when cluster actually changes
        if selected_cluster != st.session_state.get("discuss_cluster"):
            st.session_state["discuss_cluster"] = selected_cluster
            # Preserve system messages, drop old Q&A
            system_msgs = [
                m for m in st.session_state.get("chat_messages", [])
                if m["role"] == "system"
            ]
            system_msgs.append({
                "role": "assistant",
                "content": (
                    f"Switched to **{selected_cluster}**. 🔄 "
                    "Ask me anything about this career path, "
                    "skills to build, or next steps!"
                )
            })
            st.session_state["chat_messages"] = system_msgs
            st.rerun()
    else:
        selected_cluster = st.session_state.get("predicted_career", "your career")

    st.caption(f"Currently discussing: **{selected_cluster}**")

    # ── Display messages — skip system role ──
    for msg in st.session_state.get("chat_messages", []):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Quick questions ──
    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "What skills should I build next?",
        "Which companies should I target?",
        "Should I do a Masters degree?",
        "What certifications help me?",
        "How do I switch to this field?",
        "What salary can I expect?"
    ]
    for i, q in enumerate(quick_qs):
        if qcols[i % 3].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state["chat_messages"].append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                ans = rag_chat(q, st.session_state["input_dict"], selected_cluster)
            st.session_state["chat_messages"].append({"role": "assistant", "content": ans})
            st.rerun()

    # ── Free-text input ──
    user_q = st.chat_input("Ask anything about your career...")
    if user_q:
        st.session_state["chat_messages"].append({"role": "user", "content": user_q})
        with st.spinner("Thinking..."):
            ans = rag_chat(user_q, st.session_state["input_dict"], selected_cluster)
        st.session_state["chat_messages"].append({"role": "assistant", "content": ans})
        st.rerun()