"""
110+ evaluation prompts for BabyJay, grouped by category.
Each entry is (category, prompt).
"""

PROMPTS = [
    # ---------------- LEADERSHIP (5) ----------------
    ("leadership", "Who's the chair of the EECS department at KU?"),
    ("leadership", "Who is the Associate Chair for Graduate Studies in EECS?"),
    ("leadership", "Who directs I2S at KU?"),
    ("leadership", "Who leads CReSIS?"),
    ("leadership", "Who is the Associate Chair for Undergraduate Studies in EECS?"),

    # ---------------- FACULTY GENERAL (15) ----------------
    ("faculty", "Who are the EECS professors at KU?"),
    ("faculty", "Which EECS professors work on machine learning?"),
    ("faculty", "Which EECS professors work on cybersecurity?"),
    ("faculty", "Which KU faculty do research on computer vision?"),
    ("faculty", "Name some EECS professors doing robotics research at KU."),
    ("faculty", "Who teaches compilers at KU?"),
    ("faculty", "Who does research on software security at KU EECS?"),
    ("faculty", "Which EECS faculty work on radar or remote sensing?"),
    ("faculty", "Who does wireless networking research in EECS at KU?"),
    ("faculty", "Who does quantum computing research at KU?"),
    ("faculty", "Any EECS professors working on NLP?"),
    ("faculty", "EECS professors in computational biology?"),
    ("faculty", "EECS faculty who work on signal processing?"),
    ("faculty", "Which EECS faculty are Distinguished Professors?"),
    ("faculty", "How many EECS faculty are at KU total?"),

    # ---------------- SPECIFIC FACULTY (8) ----------------
    ("faculty_specific", "Tell me about Professor Arvin Agah"),
    ("faculty_specific", "Contact info for Prof Perry Alexander"),
    ("faculty_specific", "What does Prasad Kulkarni research?"),
    ("faculty_specific", "Email for Alexandru Bardas"),
    ("faculty_specific", "Office location for Bo Luo"),
    ("faculty_specific", "Who is Carl Leuschen?"),
    ("faculty_specific", "Tell me about Fengjun Li's research"),
    ("faculty_specific", "What is Drew Davidson's area of research?"),

    # ---------------- COURSES (15) ----------------
    ("courses", "What is EECS 168?"),
    ("courses", "What are the prerequisites for EECS 268?"),
    ("courses", "How many credit hours is EECS 168?"),
    ("courses", "Describe EECS 348."),
    ("courses", "What does EECS 388 cover?"),
    ("courses", "Is there a course on operating systems in EECS?"),
    ("courses", "What EECS courses cover machine learning?"),
    ("courses", "What's the course number for Discrete Structures?"),
    ("courses", "Tell me about EECS 510."),
    ("courses", "Which EECS courses cover compilers?"),
    ("courses", "Graduate EECS courses in cybersecurity?"),
    ("courses", "What is EECS 582?"),
    ("courses", "Is EECS 140 the same as EECS 168?"),
    ("courses", "What's the prerequisite chain for EECS 678?"),
    ("courses", "Which EECS courses satisfy the CS Elective requirement?"),

    # ---------------- DEGREE PROGRAMS (15) ----------------
    ("programs", "What are the requirements for a BS in Computer Science at KU?"),
    ("programs", "How many credit hours for the BS in Computer Science at KU?"),
    ("programs", "How many credit hours for BS Computer Engineering at KU?"),
    ("programs", "How many credit hours for BS Electrical Engineering at KU?"),
    ("programs", "What's the 4-year plan for BS CS at KU?"),
    ("programs", "Can I do a minor in computer science at KU?"),
    ("programs", "What's the accelerated BS/MS program in EECS?"),
    ("programs", "Does KU offer a BS in Cybersecurity?"),
    ("programs", "What is the BS in Applied Computing at KU?"),
    ("programs", "Learning outcomes for BS Computer Science at KU"),
    ("programs", "What are the core courses for BS Computer Engineering?"),
    ("programs", "Tell me about the MS in Computer Science at KU"),
    ("programs", "How is the MS in EECS structured — thesis vs non-thesis?"),
    ("programs", "How do I apply to KU's CS PhD?"),
    ("programs", "What certificates does EECS offer?"),

    # ---------------- GRAD ADMISSIONS (10) ----------------
    ("grad_admissions", "When are KU EECS graduate applications due?"),
    ("grad_admissions", "What GPA do I need for the KU CS PhD?"),
    ("grad_admissions", "Is GRE required for KU EECS grad school?"),
    ("grad_admissions", "What TOEFL score do I need for EECS at KU?"),
    ("grad_admissions", "What's the funding situation for EECS grad students?"),
    ("grad_admissions", "How many letters of recommendation for EECS grad apps?"),
    ("grad_admissions", "What are EECS deficiency courses?"),
    ("grad_admissions", "Graduate assistantships at KU EECS"),
    ("grad_admissions", "Application fee for KU EECS grad"),
    ("grad_admissions", "Can international students apply to KU EECS grad programs?"),

    # ---------------- RESEARCH / LABS (10) ----------------
    ("research", "What research clusters does KU EECS have?"),
    ("research", "Tell me about ITTC at KU"),
    ("research", "Tell me about CReSIS"),
    ("research", "What is I2S?"),
    ("research", "What research happens in EECS cybersecurity cluster?"),
    ("research", "What is the radar research at KU?"),
    ("research", "EECS computational science research"),
    ("research", "Language and semantics research at KU EECS"),
    ("research", "Signal processing research at KU EECS"),
    ("research", "Theory of computing research at KU"),

    # ---------------- STUDENT ORGS (10) ----------------
    ("student_orgs", "Is there an ACM chapter at KU?"),
    ("student_orgs", "How do I join KU's IEEE student branch?"),
    ("student_orgs", "Does KU host a hackathon?"),
    ("student_orgs", "Tell me about HackKU"),
    ("student_orgs", "Tell me about KUWIC"),
    ("student_orgs", "What tutoring is available for EECS 168?"),
    ("student_orgs", "Where does KU ACM meet?"),
    ("student_orgs", "Does KU have an HKN chapter?"),
    ("student_orgs", "Cybersecurity club at KU?"),
    ("student_orgs", "When is HackKU this year?"),

    # ---------------- FACILITIES (8) ----------------
    ("facilities", "What research labs are in Eaton Hall?"),
    ("facilities", "Where is Eaton Hall?"),
    ("facilities", "Tell me about the Computing Commons"),
    ("facilities", "What is the EECS Shop?"),
    ("facilities", "How many computer labs does EECS have?"),
    ("facilities", "Is the Computing Commons open 24/7?"),
    ("facilities", "What buildings house EECS at KU?"),
    ("facilities", "Does EECS have hardware labs?"),

    # ---------------- SCHOLARSHIPS (8) ----------------
    ("scholarships", "What scholarships are specific to EECS majors at KU?"),
    ("scholarships", "Tell me about the Garmin scholarship at KU"),
    ("scholarships", "What is the Summerfield scholarship?"),
    ("scholarships", "Is there a cybersecurity scholarship at KU?"),
    ("scholarships", "How do I apply for KU scholarships?"),
    ("scholarships", "What's UKASH at KU?"),
    ("scholarships", "What GPA do I need for the Garmin scholarship?"),
    ("scholarships", "Are there scholarships for freshmen at KU?"),

    # ---------------- ADVISING / OFFICES (5) ----------------
    ("advising", "Who advises EECS undergrads?"),
    ("advising", "How do I book an advising appointment in EECS?"),
    ("advising", "EECS department office contact info"),
    ("advising", "Where is the EECS department office?"),
    ("advising", "Phone number for the EECS office"),

    # ---------------- CAREER (5) ----------------
    ("career", "What companies recruit KU EECS students?"),
    ("career", "Does EECS have a co-op program?"),
    ("career", "Tell me about the Engineering Career Center"),
    ("career", "When is the KU Engineering Career Fair?"),
    ("career", "Where is the Engineering Career Center?"),

    # ---------------- EDGE CASES + OFF-TOPIC (10) ----------------
    ("edge_cases", "hi"),
    ("edge_cases", "who are you"),
    ("edge_cases", "history of KU"),
    ("edge_cases", "capital of France"),
    ("edge_cases", "professors"),  # ambiguous — should ask clarification
    ("edge_cases", "EECS professors doing machien lerning"),  # typo
    ("edge_cases", "Compare EECS 168 and EECS 268"),  # decomposition
    ("edge_cases", "Best CS program in the US?"),  # off-topic
    ("edge_cases", "How to prepare for a KU EECS interview?"),  # open-ended
    ("edge_cases", "What's the hardest EECS course?"),  # subjective
]

if __name__ == "__main__":
    from collections import Counter
    cats = Counter(c for c, _ in PROMPTS)
    print(f"Total prompts: {len(PROMPTS)}")
    for c, n in cats.most_common():
        print(f"  {c}: {n}")
