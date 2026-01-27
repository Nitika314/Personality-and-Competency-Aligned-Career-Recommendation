import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("career_model.joblib")
feature_columns = joblib.load("feature_columns.joblib")
numeric_medians = joblib.load("numeric_medians.joblib")
cat_modes = joblib.load("cat_modes.joblib")
risk_tolerance_mapping = joblib.load("risk_tolerance_mapping.joblib")

# CAREER ROLES DATABASE - Organized by Education Level and Field
CAREER_ROLES = {
    "Education & Social Impact": {
        "High School": {
            "Any": ["Teaching Assistant", "Childcare Worker", "Community Outreach Coordinator", "Administrative Assistant (Educational Institutions)"],
            "Computer Science": ["Educational Technology Support Specialist", "IT Support in Schools"],
            "IT": ["Educational Technology Support Specialist", "IT Support in Schools"],
            "Psychology": ["Mental Health Aide", "Youth Program Assistant"],
            "Business": ["Education Program Coordinator Assistant", "Administrative Support in NGOs"],
        },
        "Undergraduate": {
            "Any": ["Primary/Secondary School Teacher", "Social Worker", "Career Counselor", "Training Coordinator"],
            "Computer Science": ["EdTech Developer", "E-Learning Platform Coordinator", "Digital Learning Specialist"],
            "IT": ["Educational Technology Specialist", "Learning Management System Administrator"],
            "Psychology": ["School Counselor", "Community Mental Health Worker", "Behavioral Therapist"],
            "Business": ["Education Program Manager", "NGO Operations Manager", "Corporate Training Coordinator"],
            "Design": ["Instructional Designer", "Educational Content Creator"],
        },
        "Graduate": {
            "Any": ["University Professor/Lecturer", "Principal/Educational Administrator", "Curriculum Developer", "Public Policy Analyst", "Education Researcher"],
            "Computer Science": ["EdTech Research Scientist", "AI in Education Researcher", "Learning Analytics Specialist"],
            "IT": ["Chief Technology Officer (Education)", "Educational Systems Architect"],
            "Psychology": ["Licensed Clinical Psychologist", "Educational Psychologist", "Social Services Director"],
            "Business": ["NGO Executive Director", "Education Consultant", "Social Enterprise Founder"],
            "Economics": ["Education Policy Economist", "Development Program Analyst"],
        }
    },
    
    "Technology & Engineering": {
        "High School": {
            "Any": ["IT Support Technician", "Junior Web Developer", "Computer Repair Technician", "Technical Support Specialist"],
            "Computer Science": ["Junior Software Developer", "Web Developer (Entry-level)", "QA Tester"],
            "IT": ["Network Support Technician", "Help Desk Specialist", "Systems Administrator Assistant"],
            "Engineering": ["CAD Technician", "Engineering Technician", "Quality Control Technician"],
        },
        "Undergraduate": {
            "Any": ["Software Developer", "Systems Engineer", "Network Engineer", "Data Analyst"],
            "Computer Science": ["Full-Stack Developer", "Frontend/Backend Developer", "Software Engineer", "Mobile App Developer", "AI/ML Engineer"],
            "IT": ["Cloud Solutions Associate", "DevOps Engineer", "Cybersecurity Analyst", "Database Administrator"],
            "Engineering": ["Mechanical Engineer", "Civil Engineer", "Electrical Engineer", "Manufacturing Engineer"],
            "Design": ["UI/UX Developer", "Product Designer (Tech)", "Web Designer"],
            "Business": ["Technical Product Manager", "IT Business Analyst", "Technology Consultant"],
        },
        "Graduate": {
            "Any": ["Senior Software Architect", "Principal Engineer", "Research Scientist (Tech)", "Engineering Manager"],
            "Computer Science": ["Machine Learning Researcher", "AI Research Scientist", "Cloud Solutions Architect", "Tech Lead/Architect"],
            "IT": ["Chief Technology Officer", "Information Security Manager", "Enterprise Architect"],
            "Engineering": ["Lead Engineer", "R&D Director", "Engineering Consultant", "Project Manager (Engineering)"],
            "Biology": ["Bioinformatics Scientist", "Computational Biology Researcher"],
        }
    },
    
    "Design & Creative Media": {
        "High School": {
            "Any": ["Graphic Design Assistant", "Social Media Coordinator", "Content Creator (Entry-level)", "Photography Assistant"],
            "Design": ["Junior Graphic Designer", "Visual Content Creator", "Digital Artist"],
            "Computer Science": ["Web Designer (Entry-level)", "UI Design Assistant"],
        },
        "Undergraduate": {
            "Any": ["Graphic Designer", "Content Creator", "Video Editor", "Brand Designer", "Marketing Creative"],
            "Design": ["UI/UX Designer", "Product Designer", "Art Director", "Motion Graphics Designer", "Visual Designer"],
            "Computer Science": ["Frontend Designer", "Interactive Media Designer", "Game Designer"],
            "IT": ["Web Designer", "Digital Media Specialist", "UX Researcher"],
            "Business": ["Brand Strategist", "Marketing Creative Director", "Content Marketing Manager"],
            "Psychology": ["UX Researcher", "User Experience Analyst", "Human-Centered Design Specialist"],
        },
        "Graduate": {
            "Any": ["Creative Director", "Senior Art Director", "Design Strategist", "Creative Consultant"],
            "Design": ["Design Director", "Experience Design Lead", "Creative Strategy Director", "Design Research Lead"],
            "Computer Science": ["Design Technologist", "Creative Technology Director"],
            "Business": ["Chief Creative Officer", "Brand Innovation Director"],
            "Psychology": ["Design Psychology Researcher", "Behavioral Design Specialist"],
        }
    },
    
    "Healthcare & Life Sciences": {
        "High School": {
            "Any": ["Certified Nursing Assistant (CNA)", "Medical Assistant", "Pharmacy Technician", "Home Health Aide", "Patient Care Technician"],
            "Computer Science": ["Healthcare IT Support", "Medical Records Technician"],
            "IT": ["Healthcare Technology Support Specialist", "Clinical Systems Support"],
            "Biology": ["Laboratory Assistant", "Veterinary Assistant"],
        },
        "Undergraduate": {
            "Any": ["Registered Nurse (RN)", "Radiologic Technologist", "Respiratory Therapist", "Medical Laboratory Technician"],
            "Biology": ["Clinical Research Coordinator", "Medical Laboratory Scientist", "Pharmaceutical Sales Representative", "Public Health Coordinator"],
            "Computer Science": ["Health Informatics Specialist", "Medical Software Developer", "Clinical Data Analyst"],
            "IT": ["Healthcare IT Specialist", "Clinical Systems Analyst", "Telemedicine Coordinator"],
            "Psychology": ["Mental Health Counselor", "Substance Abuse Counselor", "Clinical Psychology Assistant"],
            "Business": ["Healthcare Administrator", "Hospital Operations Manager", "Health Services Manager"],
        },
        "Graduate": {
            "Any": ["Physician/Medical Doctor", "Nurse Practitioner", "Physician Assistant", "Physical Therapist", "Occupational Therapist"],
            "Biology": ["Biomedical Researcher", "Pharmacist", "Geneticist", "Microbiologist", "Epidemiologist"],
            "Computer Science": ["Bioinformatics Scientist", "Healthcare AI Researcher", "Medical Imaging Specialist"],
            "IT": ["Chief Medical Information Officer", "Healthcare Data Scientist"],
            "Psychology": ["Licensed Clinical Psychologist", "Psychiatrist", "Neuropsychologist", "Health Psychologist"],
            "Business": ["Healthcare Executive", "Hospital CEO/COO", "Healthcare Consultant"],
            "Economics": ["Healthcare Economist", "Health Policy Analyst"],
        }
    },
    
    "Business & Management": {
        "High School": {
            "Any": ["Sales Associate", "Customer Service Representative", "Administrative Assistant", "Retail Supervisor"],
            "Computer Science": ["Tech Sales Representative", "IT Business Support"],
            "Business": ["Junior Business Analyst", "Operations Assistant"],
        },
        "Undergraduate": {
            "Any": ["Business Analyst", "Project Coordinator", "Marketing Specialist", "Human Resources Specialist", "Operations Analyst"],
            "Business": ["Management Consultant", "Product Manager", "Business Development Manager", "Marketing Manager"],
            "Economics": ["Market Research Analyst", "Economic Analyst", "Financial Analyst"],
            "Computer Science": ["Technical Product Manager", "IT Project Manager", "Analytics Manager"],
            "IT": ["IT Project Manager", "Technology Consultant", "Systems Business Analyst"],
            "Psychology": ["Organizational Development Specialist", "HR Business Partner", "Talent Acquisition Manager"],
            "Engineering": ["Engineering Project Manager", "Operations Manager", "Supply Chain Analyst"],
        },
        "Graduate": {
            "Any": ["Senior Management Consultant", "Strategy Director", "Vice President (Operations)", "General Manager"],
            "Business": ["MBA Leadership Roles", "Chief Operating Officer", "Strategy Consultant (Top Firms)", "Business Unit Director"],
            "Economics": ["Chief Economist", "Strategic Planning Director", "Investment Strategy Manager"],
            "Computer Science": ["VP of Product", "Chief Product Officer", "Technology Strategy Director"],
            "Psychology": ["Chief Human Resources Officer", "Organizational Psychology Consultant"],
            "Engineering": ["VP of Operations", "Chief Operating Officer (Manufacturing)"],
        }
    },
    
    "Entrepreneurship & Freelance": {
        "High School": {
            "Any": ["Freelance Writer", "Social Media Manager", "Online Seller/E-commerce", "Gig Worker (Various)", "Independent Contractor"],
            "Computer Science": ["Freelance Web Developer", "App Developer (Indie)", "Tech Freelancer"],
            "Design": ["Freelance Graphic Designer", "Independent Artist", "Content Creator"],
        },
        "Undergraduate": {
            "Any": ["Startup Founder", "Independent Consultant", "Freelance Professional", "Small Business Owner", "E-commerce Entrepreneur"],
            "Computer Science": ["Tech Startup Founder", "SaaS Entrepreneur", "Mobile App Entrepreneur", "Freelance Software Consultant"],
            "Business": ["Business Consultant", "Social Entrepreneur", "Retail Business Owner", "Online Business Owner"],
            "Design": ["Design Agency Owner", "Creative Studio Founder", "Freelance Creative Director"],
            "IT": ["IT Consulting Business", "Managed Services Provider", "Tech Solutions Entrepreneur"],
            "Engineering": ["Engineering Consulting Firm", "Product Development Startup"],
        },
        "Graduate": {
            "Any": ["Serial Entrepreneur", "Venture-Backed Founder", "Investment Group Owner", "Large Enterprise Owner"],
            "Computer Science": ["Tech Unicorn Founder", "AI/ML Startup Founder", "Platform Business Founder"],
            "Business": ["Consulting Firm Founder", "Investment Fund Manager", "Corporate Venture Builder"],
            "Engineering": ["Deep Tech Startup Founder", "Hardware Startup Founder"],
            "Design": ["Design Innovation Lab Founder", "Experience Design Consultancy"],
        }
    },
    
    "Finance & Economics": {
        "High School": {
            "Any": ["Bank Teller", "Accounting Clerk", "Bookkeeper", "Payroll Clerk", "Financial Services Representative"],
            "Business": ["Junior Financial Analyst Assistant", "Accounting Assistant"],
            "Economics": ["Economic Research Assistant", "Data Entry Specialist (Finance)"],
        },
        "Undergraduate": {
            "Any": ["Financial Analyst", "Accountant", "Tax Associate", "Auditor", "Financial Advisor"],
            "Business": ["Investment Analyst", "Corporate Finance Associate", "Financial Planning Analyst", "Management Accountant"],
            "Economics": ["Economist", "Economic Consultant", "Policy Analyst", "Market Research Analyst"],
            "Computer Science": ["Financial Software Developer", "FinTech Analyst", "Quantitative Analyst"],
            "IT": ["Financial Systems Analyst", "Risk Technology Specialist"],
            "Engineering": ["Financial Engineer", "Risk Analyst (Technical)"],
        },
        "Graduate": {
            "Any": ["Senior Financial Manager", "Investment Banker", "Portfolio Manager", "Chief Financial Officer"],
            "Business": ["Chartered Accountant", "CFO", "Investment Banking VP", "Wealth Management Director"],
            "Economics": ["Chief Economist", "Economic Policy Director", "Macroeconomic Strategist", "Economic Research Director"],
            "Computer Science": ["Quantitative Researcher", "Algorithmic Trading Developer", "FinTech Innovation Director"],
            "Engineering": ["Financial Engineering Manager", "Risk Management Director (Quantitative)"],
        }
    }
}

# CAREER CLUSTER INFORMATION
CAREER_INFO = {
    "Education & Social Impact": {
        "description": "Transform lives and communities through education, social work, and public service. Design learning experiences, advocate for social justice, and create positive change in society through teaching, policy development, and community engagement.",
        "key_skills": ["Communication & Interpersonal Skills", "Empathy & Emotional Intelligence", "Leadership & Advocacy", "Problem Solving"],
        "salary_range": "₹3.5-12 LPA (India) | $45k-90k (US)",
        "growth_outlook": "📈 Steady Growth - 8% expected in education sector",
        "work_style": "Highly Collaborative, Structured with Purpose-Driven Focus",
        "icon": "📚",
        "color": "#50E3C2"
    },
    
    "Technology & Engineering": {
        "description": "Design, develop, and implement innovative technological solutions that shape the future. Build software systems, engineer complex infrastructure, and solve critical technical challenges across diverse industries using cutting-edge tools and methodologies.",
        "key_skills": ["Technical & Programming Skills", "Analytical & Problem Solving", "Systems Thinking", "Innovation & Adaptability"],
        "salary_range": "₹5-30 LPA (India) | $70k-180k+ (US)",
        "growth_outlook": "📈 Very High Growth - 15-20% projected in tech sectors",
        "work_style": "Individual or Collaborative, Flexible or Structured",
        "icon": "💻",
        "color": "#4A90E2"
    },
    
    "Design & Creative Media": {
        "description": "Craft compelling visual narratives and user experiences that captivate audiences. Combine artistic vision with strategic thinking to create impactful designs, engaging content, and memorable brand experiences across digital and traditional media platforms.",
        "key_skills": ["Creative & Visual Thinking", "Design Tools Proficiency", "Communication & Storytelling", "Attention to Detail"],
        "salary_range": "₹3-15 LPA (India) | $45k-100k (US)",
        "growth_outlook": "📊 Moderate Growth - 8% in creative industries",
        "work_style": "Individual or Collaborative, Highly Flexible & Creative",
        "icon": "🎨",
        "color": "#9013FE"
    },
    
    "Healthcare & Life Sciences": {
        "description": "Advance human health and scientific knowledge through clinical care, research, and biomedical innovation. Work directly with patients, conduct groundbreaking research, or develop life-saving treatments in hospitals, laboratories, and pharmaceutical companies.",
        "key_skills": ["Clinical & Scientific Knowledge", "Critical Thinking & Analysis", "Empathy & Patient Care", "Attention to Detail"],
        "salary_range": "₹4-18 LPA (India) | $60k-150k+ (US)",
        "growth_outlook": "📈 Very High Growth - 16% in healthcare services",
        "work_style": "Highly Collaborative, Structured & Protocol-Driven",
        "icon": "🏥",
        "color": "#E94B3C"
    },
    
    "Business & Management": {
        "description": "Lead organizations, optimize operations, and drive strategic growth in dynamic business environments. Analyze market trends, manage teams, develop business strategies, and make data-driven decisions that impact organizational success and profitability.",
        "key_skills": ["Leadership & People Management", "Strategic & Analytical Thinking", "Communication & Negotiation", "Business Acumen"],
        "salary_range": "₹5-25 LPA (India) | $65k-140k (US)",
        "growth_outlook": "📈 Steady Growth - 9% in management occupations",
        "work_style": "Highly Collaborative, Flexible with Results Focus",
        "icon": "💼",
        "color": "#F5A623"
    },
    
    "Entrepreneurship & Freelance": {
        "description": "Build independent ventures and create your own professional path with autonomy and innovation. Launch startups, consult independently, or operate freelance businesses that leverage your expertise while embracing the risks and rewards of self-employment.",
        "key_skills": ["Leadership & Self-Management", "Risk-Taking & Resilience", "Business Development", "Creative Problem Solving"],
        "salary_range": "Highly Variable - ₹2-50+ LPA (India) | $30k-250k+ (US)",
        "growth_outlook": "📈 High Growth - Increasing trend in gig economy",
        "work_style": "Highly Individual, Maximum Flexibility, High Risk Tolerance",
        "icon": "🚀",
        "color": "#FF6B6B"
    },
    
    "Finance & Economics": {
        "description": "Manage capital, analyze financial markets, and provide strategic economic guidance to organizations and individuals. Work with complex financial instruments, economic models, and data analytics to optimize financial performance and ensure fiscal responsibility.",
        "key_skills": ["Quantitative & Analytical Reasoning", "Financial Modeling & Analysis", "Attention to Detail", "Risk Assessment"],
        "salary_range": "₹4.5-22 LPA (India) | $60k-130k+ (US)",
        "growth_outlook": "📈 Steady Growth - 7-10% in finance sector",
        "work_style": "Individual or Collaborative, Highly Structured",
        "icon": "💰",
        "color": "#7ED321"
    }
}

 #Get career roles that match the user's education level and field of study
def get_relevant_roles(career_cluster, education_level, field_of_study):
    if career_cluster not in CAREER_ROLES:
        return ["Various professional roles in this field"]
    
    education_roles = CAREER_ROLES[career_cluster].get(education_level, {})
    
    # Get field-specific roles first
    field_roles = education_roles.get(field_of_study, [])
    
    # Add general roles
    general_roles = education_roles.get("Any", [])
    
    # Combine and remove duplicates while preserving order
    all_roles = []
    seen = set()
    
    # Prioritize field-specific roles
    for role in field_roles:
        if role not in seen:
            all_roles.append(role)
            seen.add(role)
    
    # Add general roles
    for role in general_roles:
        if role not in seen and len(all_roles) < 8:  # Limit to 8 roles
            all_roles.append(role)
            seen.add(role)
    
    return all_roles if all_roles else ["Various professional roles in this field"]

#Generate personalized explanations for why a career was recommended
def explain_prediction(input_dict, predicted_career):
    explanations = []
    # Get career data
    career_data = CAREER_INFO.get(predicted_career, {})
    # Interest-based explanations
    interests = {
        'Technology': input_dict.get('interest_technology', 0),
        'Business': input_dict.get('interest_business', 0),
        'Creative/Arts': input_dict.get('interest_creative', 0),
        'Health & Social': input_dict.get('interest_health_social', 0),
        'Research/Academic': input_dict.get('interest_research_academic', 0)
    }
    max_interest = max(interests, key=interests.get)
    max_interest_value = interests[max_interest]
    if max_interest_value >= 70:
        explanations.append(f"Your strong interest in **{max_interest}** ({max_interest_value}/100) aligns perfectly with this career path")
    elif max_interest_value >= 50:
        explanations.append(f"Your interest in **{max_interest}** ({max_interest_value}/100) matches well with this field")
    # Skills-based explanations
    skills = {
        'Technical': input_dict.get('technical_skill', 0),
        'Data Reasoning': input_dict.get('data_reasoning_skill', 0),
        'Communication': input_dict.get('communication_skill', 0),
        'Problem Solving': input_dict.get('problem_solving_skill', 0),
        'Leadership': input_dict.get('leadership_skill', 0),
        'Creative Thinking': input_dict.get('creative_thinking_skill', 0)
    }
    # Get top skills
    top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:2]
    for skill_name, skill_value in top_skills:
        if skill_value >= 4:
            explanations.append(f"Your excellent **{skill_name}** skills (rated {skill_value}/5) are highly valued in this field")
        elif skill_value >= 3:
            explanations.append(f"Your **{skill_name}** proficiency (rated {skill_value}/5) is a good foundation for this career")
    # Personality-based explanations
    personality = {
        'Openness': input_dict.get('openness', 0),
        'Conscientiousness': input_dict.get('conscientiousness', 0),
        'Extraversion': input_dict.get('extraversion', 0),
        'Agreeableness': input_dict.get('agreeableness', 0),
    }
    # Career-specific personality matches
    if predicted_career == "Technology & Engineering" and personality['Openness'] >= 4:
        explanations.append("Your high openness to new ideas suits the constantly evolving tech landscape")
    
    elif predicted_career == "Healthcare & Life Sciences" and personality['Agreeableness'] >= 4:
        explanations.append("Your compassionate nature aligns with healthcare's people-centered focus")
    
    elif predicted_career == "Business & Management" and personality['Extraversion'] >= 4:
        explanations.append("Your extraverted personality is ideal for leadership and team management roles")
    
    elif predicted_career == "Design & Creative Media" and personality['Openness'] >= 4:
        explanations.append("Your creative and open mindset is perfect for innovative design work")
    
    elif predicted_career == "Finance & Economics" and personality['Conscientiousness'] >= 4:
        explanations.append("Your organized and detail-oriented nature fits well with financial analysis")
    # Work style match
    work_style = input_dict.get('preferred_work_style', '')
    environment = input_dict.get('preferred_environment', '')
    risk = input_dict.get('risk_tolerance', 0)
    
    if work_style == 'collaborative':
        explanations.append("Your preference for **collaborative work** matches the team-oriented nature of this field")
    elif work_style == 'individual':
        explanations.append("Your preference for **independent work** aligns with the autonomous aspects of this career")
    
    if environment == 'flexible' and predicted_career in ["Design & Creative Media", "Entrepreneurship & Freelance"]:
        explanations.append("Your desire for a **flexible environment** is well-suited to this dynamic field")
    elif environment == 'structured' and predicted_career in ["Finance & Economics", "Healthcare & Life Sciences"]:
        explanations.append("Your preference for **structure** matches the organized nature of this profession")
    
    # Risk tolerance
    if risk >= 2 and predicted_career == "Entrepreneurship & Freelance":
        explanations.append("Your **high risk tolerance** is essential for entrepreneurial success")
    
    # Education and experience match
    field_of_study = input_dict.get('field_of_study', '')
    experience = input_dict.get('experience_years', 0)
    
    if field_of_study in ['Computer Science', 'IT'] and predicted_career == "Technology & Engineering":
        explanations.append(f"Your background in **{field_of_study}** provides a strong foundation for this career")
    elif field_of_study == 'Engineering' and predicted_career == "Technology & Engineering":
        explanations.append("Your engineering education directly prepares you for this field")
    elif field_of_study in ['Business', 'Economics'] and predicted_career in ["Business & Management", "Finance & Economics"]:
        explanations.append(f"Your **{field_of_study}** studies align perfectly with this career path")
    
    if experience >= 3:
        explanations.append(f"Your **{experience} years of experience** gives you a competitive advantage in this field")
    # If no explanations generated, add generic ones
    if len(explanations) == 0:
        explanations.append("Your unique combination of skills and interests makes you a good fit for this career")
        explanations.append("This field offers opportunities that match your work preferences and personality")
    return explanations

st.set_page_config(page_title="Career Recommendation", layout="wide")

st.title("🎯 Personality and Competency-Aligned Career Recommendation")
st.markdown("### Answer the questions below to get your personalized career path suggestion")
st.markdown("---")

# Form
with st.form("profile_form"):
    st.subheader("👋 Personal Information")
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        user_id = st.text_input("User ID", placeholder="Enter User ID ", help="Optional identifier for your records")
    with id_col2:
        user_name = st.text_input("Name", placeholder="Enter Name ", help="Your name (optional)")

    st.markdown("---")
    
    # Section 1: Basic Information
    st.subheader("📋 Basic Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=24)
    with col2:
        education = st.selectbox("Education Level", ["High School", "Undergraduate", "Graduate"])
    with col3:
        field = st.selectbox("Field of Study", ["Computer Science", "IT", "Engineering", "Business", "Economics","Psychology", "Design", "Biology", "Other"])
    with col4:
        experience_input = st.text_input("Years of Experience", value="2",
                                         help="Enter experience in format like '2 years' or just '2'")

    st.markdown("---")
    
    # Section 2: Personality Traits
    st.subheader("🧠 Personality Traits")
    st.caption("Rate yourself on a scale of 1 (Low) to 5 (High)")
    
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    
    with p_col1:
        openness = st.slider("Openness", min_value=1, max_value=5, value=3, 
                            help="Imagination, creativity, openness to new experiences")
    with p_col2:
        consc = st.slider("Conscientiousness", min_value=1, max_value=5, value=3,
                         help="Organization, dependability, discipline")
    with p_col3:
        extra = st.slider("Extraversion", min_value=1, max_value=5, value=3,
                         help="Sociability, assertiveness, talkativeness")
    with p_col4:
        agree = st.slider("Agreeableness", min_value=1, max_value=5, value=3,
                         help="Compassion, cooperation, trustworthiness")
    with p_col5:
        neuro = st.slider("Neuroticism", min_value=1, max_value=5, value=3,
                         help="Emotional stability, anxiety levels")

    st.markdown("---")
    
    # Section 3: Skills
    st.subheader("💪 Skills Assessment")
    st.caption("Rate your proficiency from 0 (None) to 5 (Expert)")
    
    s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)
    
    with s_col1:
        tech = st.slider("Technical", min_value=0, max_value=5, value=2)
    with s_col2:
        data = st.slider("Data Reasoning", min_value=0, max_value=5, value=2)
    with s_col3:
        comm = st.slider("Communication", min_value=0, max_value=5, value=3)
    with s_col4:
        prob = st.slider("Problem Solving", min_value=0, max_value=5, value=3)
    with s_col5:
        leader = st.slider("Leadership", min_value=0, max_value=5, value=2)
    with s_col6:
        creat = st.slider("Creative Thinking", min_value=0, max_value=5, value=3)

    st.markdown("---")
    
    # Section 4: Interests
    st.subheader("🎯 Interest Areas")
    st.caption("Rate your interest level from 0 (Not Interested) to 100 (Highly Interested)")
    
    i_col1, i_col2, i_col3, i_col4, i_col5 = st.columns(5)
    
    with i_col1:
        it = st.slider("Technology", min_value=0, max_value=100, value=50)
    with i_col2:
        bu = st.slider("Business", min_value=0, max_value=100, value=40)
    with i_col3:
        cr = st.slider("Creative/Arts", min_value=0, max_value=100, value=50)
    with i_col4:
        he = st.slider("Health & Social", min_value=0, max_value=100, value=40)
    with i_col5:
        re = st.slider("Research/Academic", min_value=0, max_value=100, value=30)

    st.markdown("---")
    
    # Section 5: Work Preferences
    st.subheader("⚙️ Work Preferences")
    
    w_col1, w_col2, w_col3 = st.columns(3)
    
    with w_col1:
        work_style = st.radio("Preferred Work Style", 
                             ["individual", "collaborative"],
                             help="Do you prefer working alone or in teams?")
    with w_col2:
        env = st.radio("Preferred Environment", 
                      ["structured", "flexible"],
                      help="Do you prefer a structured routine or flexible schedule?")
    with w_col3:
        risk = st.selectbox("Risk Tolerance", 
                           ["Low", "Medium", "High"],
                           help="How comfortable are you with uncertainty and risk?")

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Get My Career Recommendation", 
                                     type="primary", 
                                     use_container_width=True)

# Prediction
if submitted:
    
    try:
        experience = float(str(experience_input).replace("years", "").replace("year", "").strip())
    except:
        experience = 0
    
    # Ensure experience doesn't exceed age - 18
    experience = min(experience, max(0, age - 18))
    experience = max(0, experience) 
    
    # Map risk tolerance to numeric 
    risk_numeric = risk_tolerance_mapping.get(risk, 1)
    
    #Valid Fields
    valid_fields = ['Computer Science', 'Psychology', 'IT', 'Design', 'Biology', 
                'Business', 'Economics', 'Engineering', 'Other']
    input_dict = {
        "age": age,
        "experience_years": experience, 
        "openness": openness,
        "conscientiousness": consc,
        "extraversion": extra,
        "agreeableness": agree,
        "neuroticism": neuro,
        "technical_skill": tech,
        "data_reasoning_skill": data,
        "communication_skill": comm,
        "problem_solving_skill": prob,
        "leadership_skill": leader,
        "creative_thinking_skill": creat,
        "interest_technology": it,
        "interest_business": bu,
        "interest_creative": cr,
        "interest_health_social": he,
        "interest_research_academic": re,
        "risk_tolerance": risk_numeric,
        "education_level": education,
        "field_of_study": field if field in valid_fields else "Other",
        "preferred_work_style": work_style,
        "preferred_environment": env
    }

    df_input = pd.DataFrame([input_dict])

    # Filling missing Values
    numeric_features = [
        'age', 'experience_years', 'openness', 'conscientiousness', 'extraversion',
        'agreeableness', 'neuroticism', 'technical_skill', 'data_reasoning_skill',
        'communication_skill', 'problem_solving_skill', 'leadership_skill',
        'creative_thinking_skill', 'interest_technology', 'interest_business',
        'interest_creative', 'interest_health_social', 'interest_research_academic',
        'risk_tolerance'
    ]
    for col in numeric_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna(numeric_medians.get(col, df_input[col].median()))

    categorical_features = ['education_level', 'field_of_study','preferred_work_style', 'preferred_environment']
    for col in categorical_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna(cat_modes.get(col, 'Unknown'))

    # Feature Engineering
    interest_cols = ['interest_technology', 'interest_business', 'interest_creative','interest_health_social', 'interest_research_academic']
    df_input['max_interest'] = df_input[interest_cols].max(axis=1)
    df_input['tech_vs_business'] = df_input['interest_technology'] - df_input['interest_business']
    df_input['creative_vs_social'] = df_input['interest_creative'] - df_input['interest_health_social']
    df_input['tech_skill_vs_comm'] = df_input['technical_skill'] - df_input['communication_skill']
    df_input['openness_creative'] = df_input['openness'] * df_input['creative_thinking_skill']
    df_input['extraversion_lead'] = df_input['extraversion'] * df_input['leadership_skill']
    df_input['is_stem'] = df_input['field_of_study'].str.contains('computer|engineering|it|math|physics|chemistry|biology|data', case=False, na=False).astype(int)

    # Encoding
    df_encoded = pd.get_dummies(df_input, columns=categorical_features, drop_first=True)
    # Align columns with training data
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    pred = model.predict(df_aligned)[0]
    
    # Display Header with User Info
    if user_name or user_id:
        st.markdown("---")
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            if user_name:
                st.markdown(f"### 👋 Hello, **{user_name}**!")
            else:
                st.markdown("### 👋 Hello!")
        with header_col2:
            if user_id:
                st.markdown(f"**ID:** `{user_id}`")

   # Display Profile Summary
    st.markdown("---")
    st.subheader("📊 Profile Overview")
    
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.markdown("##### 👤 Demographics")
        st.info(f"""
        **Age:** {age} years  
        **Education:** {education}  
        **Field of Study:** {field}  
        **Experience:** {experience} years
        """)
        
        st.markdown("##### ⚙️ Work Preferences")
        st.info(f"""
        **Work Style:** {work_style.capitalize()}  
        **Environment:** {env.capitalize()}  
        **Risk Tolerance:** {risk}
        """)
        
        
    
    with sum_col2:
        st.markdown("##### 🧠 Personality Profile")
        personality_data = {
            "Openness": openness,
            "Conscientiousness": consc,
            "Extraversion": extra,
            "Agreeableness": agree,
            "Neuroticism": neuro
        }
        for trait, score in personality_data.items():
            st.write(f"**{trait}:** {'⭐' * score}") 
            
        st.markdown("##### 💪 Skills Overview")
        skills_data = {
            "Technical": tech,
            "Data Reasoning": data,
            "Communication": comm,
            "Problem Solving": prob,
            "Leadership": leader,
            "Creative Thinking": creat
        }
        for skill, level in skills_data.items():
            st.write(f"**{skill}:** {'⭐' * level}")
    
    with sum_col3:
        st.markdown("##### 🎯 Interest Levels")
        interests_data = {
            "Technology": it,
            "Business": bu,
            "Creative/Arts": cr,
            "Health & Social": he,
            "Research/Academic": re
        }
        
        for interest, value in interests_data.items():
            st.write(f"**{interest}:** {value}/100")
            st.progress(value / 100)
        
        
    # Display Prediction Result
    st.markdown("---")
    
    # Get career info safely
    career_data = CAREER_INFO.get(pred, {
        'icon': '🎯',
        'color': '#4A90E2',
        'description': 'This is an exciting career path with many opportunities.',
        'examples': ['Various professional roles'],
        'key_skills': ['Multiple skills required'],
        'salary_range': 'Competitive',
        'growth_outlook': 'Positive outlook',
        'work_style': 'Varies'
    })
    
    icon = career_data.get('icon', '🎯')
    color = career_data.get('color', '#4A90E2')
    
    # Header with icon and color
    st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, {color}22 0%, {color}11 100%); border-radius: 15px; margin-bottom: 20px;'>
            <h1 style='margin: 0; font-size: 3em;'>{icon}</h1>
            <h2 style='margin: 10px 0; color: {color};'>Primary Career Cluster</h2>
            <h1 style='margin: 0; color: #FFFFFF; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{pred}</h1>
        </div>
    """, unsafe_allow_html=True)

    # Career Description
    st.markdown("### 📖 About This Career Path")
    st.info(career_data.get('description', 'This career path offers exciting opportunities.'))
    
    # Examples and Details columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 💼 Example Career Roles")
        st.caption(f"Based on your education level ({education}) and field of study ({field})")
        
        # Get relevant roles using the function
        relevant_roles = get_relevant_roles(pred, education, field)
        for i, role in enumerate(relevant_roles, 1):
            st.markdown(f"{i}. **{role}**")
    
    with col2:
        st.markdown("### 📊 Career Insights")
        st.markdown("**💰 Salary Range**")
        st.write(career_data.get('salary_range', 'Competitive'))
        st.markdown("**📈 Job Market Outlook**")
        st.write(career_data.get('growth_outlook', 'Positive outlook'))
        st.markdown("**🔑 Key Skills Needed**")
        key_skills = career_data.get('key_skills', ['Multiple skills'])
        for skill in key_skills:
            st.write(f"• {skill}")
        st.markdown("**⚙️ Typical Work Style**")
        st.write(career_data.get('work_style', 'Varies'))
    
    # Why This Prediction
    st.markdown("---")
    st.markdown("### 🔍 Decision Rationale")
    st.caption("Based on your profile, here's why this career suits you:")
    explanations = explain_prediction(input_dict, pred)
    for explanation in explanations:
        st.markdown(f"✓ {explanation}")
    
 
    