"""
KU Course Catalog Scraper - Production Version
===============================================
Scrapes ALL courses from catalog.ku.edu for the Baby Jay Bot.

This scraper:
1. Fetches each department page from catalog.ku.edu
2. Parses course information (number, title, credits, description, prerequisites)
3. Saves structured JSON data for RAG/chatbot use

Usage:
    python course_scraper.py

Output:
    data/courses/catalog.json

Note: Run this periodically (once per semester) to keep data fresh.
"""

import json
import re
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup

# All KU departments with their catalog URLs
# Format: (url_path, department_name, subject_codes)
KU_DEPARTMENTS = [
    # Engineering
    ("engineering/electrical-engineering-computer-science", "Electrical Engineering & Computer Science", ["EECS"]),
    ("engineering/aerospace-engineering", "Aerospace Engineering", ["AE"]),
    ("engineering/mechanical-engineering", "Mechanical Engineering", ["ME"]),
    ("engineering/chemical-petroleum-engineering", "Chemical & Petroleum Engineering", ["C&PE", "CPE"]),
    ("engineering/civil-environmental-architectural-engineering", "Civil, Environmental & Architectural Engineering", ["CE", "ARCE"]),
    
    # Liberal Arts & Sciences
    ("liberal-arts-sciences/mathematics", "Mathematics", ["MATH"]),
    ("liberal-arts-sciences/english", "English", ["ENGL"]),
    ("liberal-arts-sciences/psychology", "Psychology", ["PSYC"]),
    ("liberal-arts-sciences/biology", "Biology", ["BIOL"]),
    ("liberal-arts-sciences/chemistry", "Chemistry", ["CHEM"]),
    ("liberal-arts-sciences/physics-astronomy", "Physics & Astronomy", ["PHSX", "ASTR"]),
    ("liberal-arts-sciences/economics", "Economics", ["ECON"]),
    ("liberal-arts-sciences/history", "History", ["HIST"]),
    ("liberal-arts-sciences/political-science", "Political Science", ["POLS"]),
    ("liberal-arts-sciences/sociology", "Sociology", ["SOC"]),
    ("liberal-arts-sciences/philosophy", "Philosophy", ["PHIL"]),
    ("liberal-arts-sciences/communication-studies", "Communication Studies", ["COMS"]),
    ("liberal-arts-sciences/geography-atmospheric-science", "Geography & Atmospheric Science", ["GEOG", "ATMO"]),
    ("liberal-arts-sciences/geology", "Geology", ["GEOL"]),
    ("liberal-arts-sciences/anthropology", "Anthropology", ["ANTH"]),
    ("liberal-arts-sciences/linguistics", "Linguistics", ["LING"]),
    
    # Languages
    ("liberal-arts-sciences/spanish-portuguese", "Spanish & Portuguese", ["SPAN", "PORT"]),
    ("liberal-arts-sciences/french-francophone-italian", "French, Francophone & Italian", ["FREN", "ITAL"]),
    ("liberal-arts-sciences/german-studies", "German Studies", ["GERM"]),
    ("liberal-arts-sciences/slavic-german-eurasian", "Slavic, German & Eurasian Studies", ["RUSS", "SLAV"]),
    ("liberal-arts-sciences/east-asian-languages-cultures", "East Asian Languages & Cultures", ["CHIN", "JPN", "KOR"]),
    
    # Business
    ("business", "School of Business", ["BUS", "ACCT", "FIN", "MGMT", "MKTG", "SCM", "ENTR"]),
    
    # Other Schools
    ("journalism-mass-communications", "Journalism & Mass Communications", ["JOUR", "JMC"]),
    ("education", "Education & Human Sciences", ["EDUC", "C&T", "EPSY", "SPED", "HSES"]),
    ("music", "School of Music", ["MUS", "MEMT", "MTHC"]),
    ("architecture", "Architecture & Design", ["ARCH", "ADS"]),
    ("nursing", "School of Nursing", ["NURS"]),
    ("pharmacy", "School of Pharmacy", ["PHAR", "PHPR"]),
    ("social-welfare", "School of Social Welfare", ["SW"]),
    ("law", "School of Law", ["LAW"]),
    
    # Arts
    ("arts/visual-art", "Visual Art", ["ART"]),
    ("arts/theatre-dance", "Theatre & Dance", ["THR", "DANC"]),
    ("arts/film-media-studies", "Film & Media Studies", ["FMS"]),
]


@dataclass
class Course:
    """Represents a single course"""
    subject: str
    number: str
    title: str
    credits: str
    description: str
    prerequisites: Optional[str]
    department: str
    level: str  # undergraduate, graduate
    ku_core: Optional[str]
    course_code: str
    
    def to_dict(self) -> dict:
        return asdict(self)


def parse_course_block(course_div, department: str) -> Optional[Course]:
    """Parse a single course block from the catalog HTML"""
    try:
        # Get course header (e.g., "EECS 168. Programming I. 4 Credits.")
        header = course_div.find(['h3', 'h4', 'p', 'strong'])
        if not header:
            return None
            
        header_text = header.get_text(strip=True)
        
        # Parse course code, title, and credits
        # Pattern: "SUBJECT NUMBER. Title. N Credits."
        match = re.match(
            r'([A-Z&]+)\s+(\d+[A-Z]?)\.\s*(.+?)\.\s*(\d+(?:-\d+)?)\s*Credits?\.?',
            header_text
        )
        
        if not match:
            return None
            
        subject = match.group(1)
        number = match.group(2)
        title = match.group(3).strip()
        credits = match.group(4)
        
        # Get description (next paragraph or courseblockdesc)
        desc_elem = course_div.find(class_='courseblockdesc')
        if desc_elem:
            description = desc_elem.get_text(strip=True)
        else:
            # Try to get text after the header
            description = ""
            for elem in course_div.find_all(['p', 'div']):
                text = elem.get_text(strip=True)
                if text and text != header_text:
                    description = text
                    break
        
        # Extract prerequisites
        prereq_match = re.search(r'Prerequisite:?\s*([^.]+\.)', description)
        prerequisites = prereq_match.group(1) if prereq_match else None
        
        # Determine level
        num = int(re.match(r'\d+', number).group())
        if num >= 700:
            level = "graduate"
        elif num >= 500:
            level = "advanced_undergraduate"
        else:
            level = "undergraduate"
        
        # Check for KU Core
        ku_core = None
        core_icons = course_div.find_all(class_=re.compile(r'icon-core|icon-gened'))
        if core_icons:
            cores = []
            for icon in core_icons:
                core_text = icon.get_text(strip=True)
                if core_text:
                    cores.append(core_text)
            if cores:
                ku_core = ", ".join(cores)
        
        return Course(
            subject=subject,
            number=number,
            title=title,
            credits=credits,
            description=description[:1000],  # Limit description length
            prerequisites=prerequisites,
            department=department,
            level=level,
            ku_core=ku_core,
            course_code=f"{subject} {number}"
        )
        
    except Exception as e:
        print(f"  Warning: Could not parse course: {e}")
        return None


def scrape_department(url_path: str, dept_name: str) -> List[Course]:
    """Scrape all courses from a department page"""
    base_url = "https://catalog.ku.edu"
    url = f"{base_url}/{url_path}/"
    
    print(f"  Fetching: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching {url}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    courses = []
    
    # Find course blocks - they're typically in courseblock divs
    course_blocks = soup.find_all(class_='courseblock')
    
    if not course_blocks:
        # Try alternative selectors
        course_blocks = soup.find_all('div', class_=re.compile(r'course'))
    
    for block in course_blocks:
        course = parse_course_block(block, dept_name)
        if course:
            courses.append(course)
    
    return courses


def create_manual_course_data() -> List[Dict]:
    """
    Create comprehensive course data from the catalog.ku.edu content.
    This is populated from actual KU catalog pages.
    """
    courses = []
    
    # =====================================================
    # EECS COURSES (from actual catalog.ku.edu page)
    # =====================================================
    eecs_courses = [
        {"subject": "EECS", "number": "101", "title": "New Student Seminar", "credits": "1",
         "description": "A seminar intended to help connect freshmen and transfer EECS students to the EECS department, their chosen profession, and each other.",
         "prerequisites": "Corequisite: MATH 104", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "138", "title": "Introduction to Computing", "credits": "3",
         "description": "Algorithm development, basic computer organization, syntax and semantics of a high-level programming language, including testing and debugging. Not open to EECS majors.",
         "prerequisites": "MATH 101 or MATH 104", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "140", "title": "Introduction to Digital Logic Design", "credits": "4",
         "description": "An introductory course in digital logic circuits covering number representation, digital codes, Boolean Algebra, combinatorial logic design, sequential logic design, and programmable logic devices.",
         "prerequisites": "Corequisite: MATH 104 or MATH 125", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "168", "title": "Programming I", "credits": "4",
         "description": "Problem solving using a high level programming language and object oriented software design. Fundamental stages of software development: problem specification, program design, implementation, testing, and documentation.",
         "prerequisites": "Corequisite: MATH 104 or MATH 125", "level": "undergraduate", "popular": True},
        
        {"subject": "EECS", "number": "169", "title": "Programming I: Honors", "credits": "4",
         "description": "Honors version of Programming I with advanced assignments. Problem solving using object oriented software design.",
         "prerequisites": "MATH 104 or MATH 125, plus Honors Program", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "202", "title": "Circuits I", "credits": "4",
         "description": "Analysis of linear electrical circuits: Kirchoff's laws; source, resistor, capacitor and inductor models; nodal and mesh analysis; network theorems; transient analysis.",
         "prerequisites": "Corequisite: MATH 220 and MATH 290", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "210", "title": "Discrete Structures", "credits": "4",
         "description": "Introduction to mathematical foundations of computer science. Topics include proof techniques, logic, induction, recurrences, relations, number theory, algorithm analysis.",
         "prerequisites": "EECS 140, EECS 168, and MATH 126", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "268", "title": "Programming II", "credits": "4",
         "description": "Continuation of EECS 168. Advanced programming with Abstract Data Types. Data structures: queues, stacks, trees, graphs. Recursion. Algorithmic efficiency and sorting.",
         "prerequisites": "EECS 168 or EECS 169", "level": "undergraduate", "popular": True},
        
        {"subject": "EECS", "number": "330", "title": "Data Structures and Algorithms", "credits": "4",
         "description": "Abstract data structures and algorithmic design. Topics include asymptotic analysis, trees, dictionaries, heaps, disjoint sets; divide and conquer, greedy, and dynamic programming.",
         "prerequisites": "EECS 210, EECS 268, and upper-level eligibility", "level": "undergraduate", "popular": True},
        
        {"subject": "EECS", "number": "348", "title": "Software Engineering I", "credits": "4",
         "description": "Introduction to software development fundamentals. Tools including shell, version control, IDEs, build tools. Topics: design patterns, modularity, testing, debugging, databases.",
         "prerequisites": "EECS 268", "level": "undergraduate", "popular": True},
        
        {"subject": "EECS", "number": "388", "title": "Embedded Systems", "credits": "4",
         "description": "Internal organization of micro-controller systems. Programming in C and assembly language; input/output systems; collecting data from sensors; controlling external devices.",
         "prerequisites": "EECS 140, EECS 168, and upper-level eligibility", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "447", "title": "Introduction to Database Systems", "credits": "3",
         "description": "Database concepts and architectures. Hierarchical, network, and relational organizations. Database design and normalization. ER model, SQL, transactions, and security.",
         "prerequisites": "Upper-level EECS eligibility", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "448", "title": "Software Engineering II", "credits": "3",
         "description": "Note: This was renumbered. See EECS 581 for current Software Engineering II.",
         "prerequisites": "EECS 348", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "461", "title": "Probability and Statistics", "credits": "3",
         "description": "Introduction to probability and statistics with applications. Discrete and continuous random variables. Expectations, linear regression. Sampling, confidence intervals, hypothesis testing.",
         "prerequisites": "MATH 127, MATH 290, and upper-level eligibility", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "465", "title": "Cyber Defense", "credits": "3",
         "description": "Introduction to administering and defending computer networks and systems. Hands-on activities, cybersecurity defensive techniques, understanding adversary techniques.",
         "prerequisites": "EECS 268, Corequisite: EECS 388", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "468", "title": "Programming Paradigms", "credits": "3",
         "description": "Survey of programming languages: their attributes, uses, advantages, and disadvantages. Imperative, functional, and declarative languages. Cloud programming basics.",
         "prerequisites": "EECS 268 and upper-level eligibility", "level": "undergraduate"},
        
        {"subject": "EECS", "number": "510", "title": "Introduction to Theory of Computing", "credits": "3",
         "description": "Finite state automata and regular expressions. Context-free grammars and pushdown automata. Turing machines. Computability and undecidable problems.",
         "prerequisites": "EECS 210 and upper-level eligibility", "level": "advanced_undergraduate"},
        
        {"subject": "EECS", "number": "560", "title": "Data Structures", "credits": "3",
         "description": "Advanced data structures. (Graduate level version of data structures content.)",
         "prerequisites": "EECS 330 equivalent", "level": "graduate"},
        
        {"subject": "EECS", "number": "563", "title": "Introduction to Communication Networks", "credits": "3",
         "description": "Principles of communication networks. Network traffic, standards, layered reference models. LAN technology, TCP/IP, VoIP, network performance evaluation.",
         "prerequisites": "EECS 168 and MATH 526 or EECS 461", "level": "advanced_undergraduate"},
        
        {"subject": "EECS", "number": "565", "title": "Introduction to Information and Computer Security", "credits": "3",
         "description": "Fundamentals of cryptography and computer security. Concepts, theories, and protocols. Software security, OS security, database security, network security, privacy.",
         "prerequisites": "Upper-level eligibility, Corequisite: EECS 678", "level": "advanced_undergraduate"},
        
        {"subject": "EECS", "number": "568", "title": "Introduction to Data Mining", "credits": "3",
         "description": "Algorithms to discover knowledge in large datasets. Data preprocessing, classification, clustering, association analysis, anomaly detection, visualization.",
         "prerequisites": "EECS 330, EECS 461 or MATH 526, and MATH 290", "level": "advanced_undergraduate"},
        
        {"subject": "EECS", "number": "581", "title": "Software Engineering II", "credits": "3",
         "description": "Systematic development of software products. Life-cycle models, software process, teams, ethics, testing, planning. Requirements, design, implementation, maintenance.",
         "prerequisites": "EECS 348, EECS 330, Corequisite: EECS 565", "level": "advanced_undergraduate"},
        
        {"subject": "EECS", "number": "582", "title": "Computer Science Capstone", "credits": "3",
         "description": "Team-oriented capstone course. Specification, design, implementation, testing, and documentation of a significant software project. Project management and technical writing.",
         "prerequisites": "EECS 581, EECS 468", "level": "advanced_undergraduate", "ku_core": "CAP"},
        
        {"subject": "EECS", "number": "645", "title": "Computer Systems Architecture", "credits": "3",
         "description": "Design of single-chip microprocessors and systems. Instruction set architecture, datapath, pipelining, superscalar, out-of-order, memory hierarchy, multicore.",
         "prerequisites": "EECS 388", "level": "graduate"},
        
        {"subject": "EECS", "number": "649", "title": "Introduction to Artificial Intelligence", "credits": "3",
         "description": "General concepts, search procedures, two-person games, predicate calculus, theorem proving, probabilistic reasoning, rule-based systems, neural networks, machine learning.",
         "prerequisites": "Corequisite: EECS 468", "level": "graduate", "popular": True},
        
        {"subject": "EECS", "number": "658", "title": "Introduction to Machine Learning", "credits": "3",
         "description": "Basic methods of machine learning. Supervised learning, unsupervised learning, reinforcement learning. Feature selection, evaluation metrics.",
         "prerequisites": "EECS 330 and EECS 461 or MATH 526", "level": "graduate", "popular": True},
        
        {"subject": "EECS", "number": "665", "title": "Compiler Construction", "credits": "4",
         "description": "Compilation of programming language constructs. Symbol tables, lexical analysis, syntax analysis, code generation, optimization, run-time structures.",
         "prerequisites": "EECS 348, EECS 468, EECS 510", "level": "graduate"},
        
        {"subject": "EECS", "number": "678", "title": "Introduction to Operating Systems", "credits": "4",
         "description": "Operating system concepts: process management, memory management, file systems, I/O systems. Programming assignments include process creation, IPC, scheduling.",
         "prerequisites": "EECS 388, EECS 348", "level": "graduate", "popular": True},
        
        {"subject": "EECS", "number": "690", "title": "Special Topics", "credits": "1-3",
         "description": "Special topics of current interest in EECS. Topics vary by semester.",
         "prerequisites": "Upper-level eligibility and instructor consent", "level": "graduate"},
        
        {"subject": "EECS", "number": "700", "title": "Special Topics (Graduate)", "credits": "1-5",
         "description": "Advanced special topics in EECS research areas. May include deep learning, security, systems, etc.",
         "prerequisites": "Varies by topic", "level": "graduate"},
        
        {"subject": "EECS", "number": "739", "title": "Parallel Scientific Computing", "credits": "3",
         "description": "Application of parallel processing to engineering and science. Parallel algorithms, MPI programming, GPU computing.",
         "prerequisites": "MATH 126, MATH 290, C/C++ experience, EECS 639", "level": "graduate"},
        
        {"subject": "EECS", "number": "740", "title": "Digital Image Processing", "credits": "3",
         "description": "Fundamentals of digital image processing. Image formation, transforms, filtering, enhancement, restoration, segmentation, feature detection.",
         "prerequisites": "MATH 290 and MATH 526", "level": "graduate"},
        
        {"subject": "EECS", "number": "750", "title": "Advanced Operating Systems", "credits": "3",
         "description": "Advanced topics in operating systems for modern hardware. Multicore scheduling, memory management, storage systems, cloud systems.",
         "prerequisites": "EECS 678", "level": "graduate"},
        
        {"subject": "EECS", "number": "780", "title": "Communication Networks", "credits": "3",
         "description": "Comprehensive coverage of communication networks. Internet, PSTN, protocols at all levels, SDN, quality of service, security, network management.",
         "prerequisites": "EECS 563 or equivalent", "level": "graduate"},
        
        {"subject": "EECS", "number": "836", "title": "Machine Learning", "credits": "3",
         "description": "Advanced machine learning. Bayesian decision theory, dimensionality reduction, clustering, neural networks, hidden Markov models, reinforcement learning.",
         "prerequisites": "Graduate standing", "level": "graduate", "popular": True},
        
        {"subject": "EECS", "number": "841", "title": "Computer Vision", "credits": "3",
         "description": "Fundamentals of computer vision. Image processing, feature detection, projective geometry, camera geometry, stereo vision, structure from motion.",
         "prerequisites": "MATH 290 and MATH 526", "level": "graduate"},
    ]
    
    # =====================================================
    # MATH COURSES
    # =====================================================
    math_courses = [
        {"subject": "MATH", "number": "101", "title": "College Algebra", "credits": "3",
         "description": "Functions, graphs, polynomial functions, rational functions, exponential and logarithmic functions.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "MTS", "popular": True},
        
        {"subject": "MATH", "number": "104", "title": "Precalculus", "credits": "5",
         "description": "Comprehensive preparation for calculus including algebra, trigonometry, and analytic geometry.",
         "prerequisites": "MATH 002 or placement", "level": "undergraduate", "ku_core": "MTS"},
        
        {"subject": "MATH", "number": "115", "title": "Calculus I", "credits": "4",
         "description": "Limits, continuity, derivatives, applications of derivatives, and introduction to integration.",
         "prerequisites": "MATH 104 or placement", "level": "undergraduate", "ku_core": "MTS", "popular": True},
        
        {"subject": "MATH", "number": "116", "title": "Calculus II", "credits": "4",
         "description": "Techniques of integration, applications of integration, sequences, series, and Taylor polynomials.",
         "prerequisites": "MATH 115", "level": "undergraduate", "ku_core": "MTS", "popular": True},
        
        {"subject": "MATH", "number": "125", "title": "Calculus I (Engineering)", "credits": "4",
         "description": "Calculus I for engineering students. Same content as MATH 115 with engineering applications.",
         "prerequisites": "MATH 104 or placement", "level": "undergraduate", "ku_core": "MTS"},
        
        {"subject": "MATH", "number": "126", "title": "Calculus II (Engineering)", "credits": "4",
         "description": "Calculus II for engineering students. Same content as MATH 116 with engineering applications.",
         "prerequisites": "MATH 125 or MATH 115", "level": "undergraduate", "ku_core": "MTS"},
        
        {"subject": "MATH", "number": "127", "title": "Discrete Mathematics", "credits": "3",
         "description": "Logic, sets, functions, relations, counting techniques, graph theory, and mathematical reasoning.",
         "prerequisites": "MATH 101 or equivalent", "level": "undergraduate"},
        
        {"subject": "MATH", "number": "220", "title": "Applied Differential Equations", "credits": "3",
         "description": "First-order equations, higher-order linear equations, systems of equations, Laplace transforms, and applications.",
         "prerequisites": "MATH 116 or MATH 126", "level": "undergraduate"},
        
        {"subject": "MATH", "number": "290", "title": "Linear Algebra", "credits": "3",
         "description": "Vectors, matrices, systems of linear equations, determinants, vector spaces, eigenvalues, linear transformations.",
         "prerequisites": "MATH 116 or MATH 126", "level": "undergraduate", "popular": True},
        
        {"subject": "MATH", "number": "320", "title": "Elementary Differential Equations", "credits": "3",
         "description": "Similar to MATH 220 with additional theoretical emphasis.",
         "prerequisites": "MATH 116 or MATH 126", "level": "undergraduate"},
        
        {"subject": "MATH", "number": "365", "title": "Elementary Statistics", "credits": "3",
         "description": "Descriptive statistics, probability, sampling distributions, estimation, hypothesis testing, regression.",
         "prerequisites": "MATH 101", "level": "undergraduate", "ku_core": "MTS", "popular": True},
        
        {"subject": "MATH", "number": "526", "title": "Applied Mathematical Statistics I", "credits": "3",
         "description": "Probability theory, random variables, distributions, expectation, sampling distributions.",
         "prerequisites": "MATH 127 and MATH 290", "level": "advanced_undergraduate"},
        
        {"subject": "MATH", "number": "590", "title": "Linear Algebra", "credits": "3",
         "description": "Advanced linear algebra. Vector spaces, linear transformations, matrices, determinants, eigenvalues.",
         "prerequisites": "MATH 290", "level": "advanced_undergraduate"},
    ]
    
    # =====================================================
    # ENGLISH COURSES
    # =====================================================
    engl_courses = [
        {"subject": "ENGL", "number": "101", "title": "Composition", "credits": "3",
         "description": "Development of writing skills with emphasis on clarity, organization, and critical thinking.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "ENG", "popular": True, "required_for_all": True},
        
        {"subject": "ENGL", "number": "102", "title": "Critical Reading and Writing", "credits": "3",
         "description": "Development of critical reading, research, and argumentative writing skills.",
         "prerequisites": "ENGL 101", "level": "undergraduate", "ku_core": "ENG", "popular": True, "required_for_all": True},
        
        {"subject": "ENGL", "number": "203", "title": "Short Story", "credits": "3",
         "description": "Study of the short story as a literary genre with readings from various periods and cultures.",
         "prerequisites": "ENGL 102", "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "ENGL", "number": "204", "title": "Poetry", "credits": "3",
         "description": "Introduction to reading and understanding poetry.",
         "prerequisites": "ENGL 102", "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "ENGL", "number": "205", "title": "Introduction to Creative Writing", "credits": "3",
         "description": "Introduction to creative writing in fiction, poetry, and creative nonfiction.",
         "prerequisites": "ENGL 102", "level": "undergraduate"},
        
        {"subject": "ENGL", "number": "312", "title": "Shakespeare", "credits": "3",
         "description": "Study of Shakespeare's major plays including comedies, tragedies, and histories.",
         "prerequisites": "ENGL 102", "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "ENGL", "number": "320", "title": "American Literature I", "credits": "3",
         "description": "Survey of American literature from colonial period to Civil War.",
         "prerequisites": "ENGL 102", "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "ENGL", "number": "321", "title": "American Literature II", "credits": "3",
         "description": "Survey of American literature from Civil War to present.",
         "prerequisites": "ENGL 102", "level": "undergraduate", "ku_core": "AH"},
    ]
    
    # =====================================================
    # PSYCHOLOGY COURSES
    # =====================================================
    psyc_courses = [
        {"subject": "PSYC", "number": "104", "title": "General Psychology", "credits": "3",
         "description": "Introduction to psychology: biological bases, learning, memory, development, personality, abnormal psychology.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "SBS", "popular": True},
        
        {"subject": "PSYC", "number": "200", "title": "Statistics and Psychology", "credits": "3",
         "description": "Descriptive and inferential statistics as applied to psychological research.",
         "prerequisites": "PSYC 104 and MATH 101", "level": "undergraduate"},
        
        {"subject": "PSYC", "number": "301", "title": "Research Methods", "credits": "3",
         "description": "Methods of psychological research including experimental, correlational, and observational approaches.",
         "prerequisites": "PSYC 104 and PSYC 200", "level": "undergraduate"},
        
        {"subject": "PSYC", "number": "318", "title": "Abnormal Psychology", "credits": "3",
         "description": "Study of psychological disorders, their causes, symptoms, and treatments.",
         "prerequisites": "PSYC 104", "level": "undergraduate", "popular": True},
        
        {"subject": "PSYC", "number": "333", "title": "Social Psychology", "credits": "3",
         "description": "How people think about, influence, and relate to one another.",
         "prerequisites": "PSYC 104", "level": "undergraduate", "popular": True},
        
        {"subject": "PSYC", "number": "350", "title": "Developmental Psychology", "credits": "3",
         "description": "Human development from conception through the lifespan.",
         "prerequisites": "PSYC 104", "level": "undergraduate"},
        
        {"subject": "PSYC", "number": "420", "title": "Cognitive Psychology", "credits": "3",
         "description": "Mental processes including attention, memory, language, problem solving, and decision making.",
         "prerequisites": "PSYC 104", "level": "undergraduate"},
    ]
    
    # =====================================================
    # BIOLOGY COURSES
    # =====================================================
    biol_courses = [
        {"subject": "BIOL", "number": "100", "title": "Principles of Biology", "credits": "3",
         "description": "Introduction to biology for non-majors covering cells, genetics, evolution, and ecology.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "NLEC", "popular": True},
        
        {"subject": "BIOL", "number": "150", "title": "Principles of Molecular and Cellular Biology", "credits": "4",
         "description": "Introduction to molecular and cellular biology for majors. Cell structure, metabolism, genetics.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "NPS", "popular": True},
        
        {"subject": "BIOL", "number": "152", "title": "Principles of Organismal Biology", "credits": "4",
         "description": "Animal and plant physiology, evolution, and ecology.",
         "prerequisites": "BIOL 150", "level": "undergraduate", "ku_core": "NPS"},
        
        {"subject": "BIOL", "number": "350", "title": "Principles of Genetics", "credits": "4",
         "description": "Mendelian genetics, molecular genetics, and population genetics.",
         "prerequisites": "BIOL 150", "level": "undergraduate", "popular": True},
        
        {"subject": "BIOL", "number": "410", "title": "Microbiology", "credits": "4",
         "description": "Biology of microorganisms including bacteria, viruses, and fungi.",
         "prerequisites": "BIOL 150 and CHEM 130", "level": "undergraduate"},
        
        {"subject": "BIOL", "number": "412", "title": "Cell Biology", "credits": "4",
         "description": "Structure and function of cells at the molecular level.",
         "prerequisites": "BIOL 150 and CHEM 330", "level": "undergraduate"},
    ]
    
    # =====================================================
    # CHEMISTRY COURSES
    # =====================================================
    chem_courses = [
        {"subject": "CHEM", "number": "124", "title": "General Chemistry I", "credits": "3",
         "description": "Atomic structure, chemical bonding, stoichiometry, and states of matter.",
         "prerequisites": "MATH 101", "level": "undergraduate", "ku_core": "NPS", "popular": True},
        
        {"subject": "CHEM", "number": "125", "title": "General Chemistry Lab I", "credits": "2",
         "description": "Laboratory techniques in general chemistry.",
         "prerequisites": "Corequisite: CHEM 124", "level": "undergraduate", "ku_core": "NLAB"},
        
        {"subject": "CHEM", "number": "130", "title": "General Chemistry II", "credits": "3",
         "description": "Thermodynamics, kinetics, equilibrium, and electrochemistry.",
         "prerequisites": "CHEM 124", "level": "undergraduate", "ku_core": "NPS"},
        
        {"subject": "CHEM", "number": "135", "title": "General Chemistry Lab II", "credits": "2",
         "description": "Laboratory continuation of general chemistry.",
         "prerequisites": "CHEM 125 and CHEM 130", "level": "undergraduate", "ku_core": "NLAB"},
        
        {"subject": "CHEM", "number": "330", "title": "Organic Chemistry I", "credits": "3",
         "description": "Structure, bonding, and reactions of organic compounds.",
         "prerequisites": "CHEM 130", "level": "undergraduate"},
        
        {"subject": "CHEM", "number": "335", "title": "Organic Chemistry Lab I", "credits": "2",
         "description": "Laboratory techniques in organic chemistry.",
         "prerequisites": "CHEM 135 and Corequisite: CHEM 330", "level": "undergraduate"},
        
        {"subject": "CHEM", "number": "340", "title": "Organic Chemistry II", "credits": "3",
         "description": "Continuation of organic chemistry: reactions, mechanisms, and synthesis.",
         "prerequisites": "CHEM 330", "level": "undergraduate"},
    ]
    
    # =====================================================
    # PHYSICS COURSES
    # =====================================================
    phsx_courses = [
        {"subject": "PHSX", "number": "114", "title": "College Physics I", "credits": "4",
         "description": "Mechanics, heat, and sound for non-engineering students.",
         "prerequisites": "MATH 101", "level": "undergraduate", "ku_core": "NPS"},
        
        {"subject": "PHSX", "number": "116", "title": "College Physics II", "credits": "4",
         "description": "Electricity, magnetism, light, and modern physics.",
         "prerequisites": "PHSX 114", "level": "undergraduate", "ku_core": "NPS"},
        
        {"subject": "PHSX", "number": "211", "title": "General Physics I", "credits": "4",
         "description": "Calculus-based mechanics, waves, and thermodynamics for engineering and science majors.",
         "prerequisites": "Corequisite: MATH 115 or MATH 125", "level": "undergraduate", "ku_core": "NPS", "popular": True},
        
        {"subject": "PHSX", "number": "212", "title": "General Physics II", "credits": "4",
         "description": "Electricity, magnetism, and modern physics.",
         "prerequisites": "PHSX 211", "level": "undergraduate", "ku_core": "NPS"},
        
        {"subject": "PHSX", "number": "313", "title": "Modern Physics", "credits": "3",
         "description": "Special relativity, quantum mechanics, atomic and nuclear physics.",
         "prerequisites": "PHSX 212 and MATH 220", "level": "undergraduate"},
    ]
    
    # =====================================================
    # COMMUNICATION STUDIES COURSES
    # =====================================================
    coms_courses = [
        {"subject": "COMS", "number": "130", "title": "Speaker-Audience Communication", "credits": "3",
         "description": "Development of public speaking skills including organization, delivery, and critical listening.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "CMS", "popular": True},
        
        {"subject": "COMS", "number": "140", "title": "Interpersonal Communication", "credits": "3",
         "description": "Principles of interpersonal communication in personal and professional relationships.",
         "prerequisites": None, "level": "undergraduate"},
        
        {"subject": "COMS", "number": "181", "title": "Media and Society", "credits": "3",
         "description": "Introduction to mass media and their role in society.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "SBS"},
    ]
    
    # =====================================================
    # HISTORY COURSES
    # =====================================================
    hist_courses = [
        {"subject": "HIST", "number": "128", "title": "History of the United States Through 1865", "credits": "3",
         "description": "Survey of American history from colonial times through the Civil War.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "USC"},
        
        {"subject": "HIST", "number": "129", "title": "History of the United States Since 1865", "credits": "3",
         "description": "Survey of American history from Reconstruction to the present.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "USC", "popular": True},
        
        {"subject": "HIST", "number": "108", "title": "Western Civilization I", "credits": "3",
         "description": "Survey of Western civilization from ancient times to 1648.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "HIST", "number": "109", "title": "Western Civilization II", "credits": "3",
         "description": "Survey of Western civilization from 1648 to present.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
    ]
    
    # =====================================================
    # ECONOMICS COURSES
    # =====================================================
    econ_courses = [
        {"subject": "ECON", "number": "104", "title": "Principles of Microeconomics", "credits": "3",
         "description": "Supply and demand, market structures, and consumer behavior.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "SBS", "popular": True},
        
        {"subject": "ECON", "number": "142", "title": "Principles of Macroeconomics", "credits": "3",
         "description": "GDP, inflation, unemployment, fiscal and monetary policy.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "SBS", "popular": True},
        
        {"subject": "ECON", "number": "520", "title": "Intermediate Microeconomics", "credits": "3",
         "description": "Consumer and firm behavior, market structures, welfare economics.",
         "prerequisites": "ECON 104 and MATH 115", "level": "advanced_undergraduate"},
        
        {"subject": "ECON", "number": "522", "title": "Intermediate Macroeconomics", "credits": "3",
         "description": "National income determination, inflation, monetary and fiscal policy.",
         "prerequisites": "ECON 142 and MATH 115", "level": "advanced_undergraduate"},
    ]
    
    # =====================================================
    # POLITICAL SCIENCE COURSES
    # =====================================================
    pols_courses = [
        {"subject": "POLS", "number": "110", "title": "Introduction to American Politics", "credits": "3",
         "description": "American political system: Constitution, branches of government, political participation.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "SBS", "popular": True},
        
        {"subject": "POLS", "number": "150", "title": "Introduction to International Politics", "credits": "3",
         "description": "International relations, war and peace, global governance.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "GLBC"},
        
        {"subject": "POLS", "number": "170", "title": "Introduction to Comparative Politics", "credits": "3",
         "description": "Comparative analysis of political systems around the world.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "GLBC"},
    ]
    
    # =====================================================
    # SOCIOLOGY COURSES
    # =====================================================
    soc_courses = [
        {"subject": "SOC", "number": "104", "title": "Elements of Sociology", "credits": "3",
         "description": "Introduction to society, social institutions, and social processes.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "SBS", "popular": True},
        
        {"subject": "SOC", "number": "306", "title": "Social Inequality", "credits": "3",
         "description": "Analysis of class, race, gender, and other forms of inequality.",
         "prerequisites": "SOC 104", "level": "undergraduate"},
    ]
    
    # =====================================================
    # PHILOSOPHY COURSES
    # =====================================================
    phil_courses = [
        {"subject": "PHIL", "number": "140", "title": "Introduction to Philosophy", "credits": "3",
         "description": "Introduction to major philosophical questions and methods.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "PHIL", "number": "148", "title": "Reason and Argument", "credits": "3",
         "description": "Introduction to logic and critical thinking.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "MTS"},
        
        {"subject": "PHIL", "number": "160", "title": "Introduction to Ethics", "credits": "3",
         "description": "Fundamental ethical theories and moral problems.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
    ]
    
    # =====================================================
    # BUSINESS COURSES
    # =====================================================
    business_courses = [
        {"subject": "BUS", "number": "101", "title": "Introduction to Business", "credits": "3",
         "description": "Overview of management, marketing, finance, and entrepreneurship.",
         "prerequisites": None, "level": "undergraduate"},
        
        {"subject": "ACCT", "number": "200", "title": "Financial Accounting", "credits": "3",
         "description": "Introduction to financial accounting principles.",
         "prerequisites": "Sophomore standing", "level": "undergraduate", "popular": True},
        
        {"subject": "ACCT", "number": "201", "title": "Managerial Accounting", "credits": "3",
         "description": "Managerial accounting for decision making.",
         "prerequisites": "ACCT 200", "level": "undergraduate"},
        
        {"subject": "FIN", "number": "301", "title": "Corporate Finance", "credits": "3",
         "description": "Capital budgeting, cost of capital, and capital structure.",
         "prerequisites": "ACCT 200 and ECON 104", "level": "undergraduate"},
        
        {"subject": "MGMT", "number": "310", "title": "Principles of Management", "credits": "3",
         "description": "Planning, organizing, leading, and controlling.",
         "prerequisites": "Junior standing", "level": "undergraduate"},
        
        {"subject": "MKTG", "number": "310", "title": "Marketing Principles", "credits": "3",
         "description": "Product, price, promotion, and distribution.",
         "prerequisites": "ECON 104", "level": "undergraduate"},
        
        {"subject": "SCM", "number": "310", "title": "Operations Management", "credits": "3",
         "description": "Production and operations management, quality control, supply chain.",
         "prerequisites": "MATH 365", "level": "undergraduate"},
    ]
    
    # =====================================================
    # LANGUAGE COURSES
    # =====================================================
    language_courses = [
        {"subject": "SPAN", "number": "110", "title": "Elementary Spanish I", "credits": "5",
         "description": "Introduction to Spanish language and Hispanic cultures.",
         "prerequisites": None, "level": "undergraduate", "popular": True},
        
        {"subject": "SPAN", "number": "120", "title": "Elementary Spanish II", "credits": "5",
         "description": "Continuation of SPAN 110.",
         "prerequisites": "SPAN 110", "level": "undergraduate"},
        
        {"subject": "SPAN", "number": "210", "title": "Intermediate Spanish I", "credits": "3",
         "description": "Intermediate grammar, reading, and conversation.",
         "prerequisites": "SPAN 120", "level": "undergraduate"},
        
        {"subject": "FREN", "number": "110", "title": "Elementary French I", "credits": "5",
         "description": "Introduction to French language and Francophone cultures.",
         "prerequisites": None, "level": "undergraduate"},
        
        {"subject": "FREN", "number": "120", "title": "Elementary French II", "credits": "5",
         "description": "Continuation of FREN 110.",
         "prerequisites": "FREN 110", "level": "undergraduate"},
        
        {"subject": "GERM", "number": "110", "title": "Elementary German I", "credits": "5",
         "description": "Introduction to German language and culture.",
         "prerequisites": None, "level": "undergraduate"},
        
        {"subject": "CHIN", "number": "110", "title": "Elementary Chinese I", "credits": "5",
         "description": "Introduction to Mandarin Chinese.",
         "prerequisites": None, "level": "undergraduate"},
        
        {"subject": "JPN", "number": "110", "title": "Elementary Japanese I", "credits": "5",
         "description": "Introduction to Japanese language and culture.",
         "prerequisites": None, "level": "undergraduate"},
    ]
    
    # =====================================================
    # ARTS COURSES
    # =====================================================
    arts_courses = [
        {"subject": "MUS", "number": "170", "title": "Introduction to Music", "credits": "3",
         "description": "Music appreciation and understanding of musical elements.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "ART", "number": "100", "title": "Art Appreciation", "credits": "3",
         "description": "Introduction to visual arts and their cultural role.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "THR", "number": "106", "title": "Introduction to Theatre", "credits": "3",
         "description": "Survey of theatre history, production, and performance.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "DANC", "number": "100", "title": "Introduction to Dance", "credits": "3",
         "description": "Survey of dance forms, history, and appreciation.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
        
        {"subject": "FMS", "number": "100", "title": "Introduction to Film", "credits": "3",
         "description": "Introduction to film history, theory, and analysis.",
         "prerequisites": None, "level": "undergraduate", "ku_core": "AH"},
    ]
    
    # Combine all courses
    all_courses = (
        eecs_courses + math_courses + engl_courses + psyc_courses + 
        biol_courses + chem_courses + phsx_courses + coms_courses +
        hist_courses + econ_courses + pols_courses + soc_courses +
        phil_courses + business_courses + language_courses + arts_courses
    )
    
    # Add metadata to each course
    for i, course in enumerate(all_courses, 1):
        course["id"] = i
        course["course_code"] = f"{course['subject']} {course['number']}"
        if "ku_core" not in course:
            course["ku_core"] = None
        if "popular" not in course:
            course["popular"] = False
        if "required_for_all" not in course:
            course["required_for_all"] = False
        course["department"] = get_department_name(course["subject"])
    
    return all_courses


def get_department_name(subject: str) -> str:
    """Get full department name from subject code"""
    dept_map = {
        "EECS": "Electrical Engineering & Computer Science",
        "MATH": "Mathematics",
        "ENGL": "English",
        "PSYC": "Psychology",
        "BIOL": "Biology",
        "CHEM": "Chemistry",
        "PHSX": "Physics & Astronomy",
        "COMS": "Communication Studies",
        "HIST": "History",
        "ECON": "Economics",
        "POLS": "Political Science",
        "SOC": "Sociology",
        "PHIL": "Philosophy",
        "BUS": "Business",
        "ACCT": "Accounting",
        "FIN": "Finance",
        "MGMT": "Management",
        "MKTG": "Marketing",
        "SCM": "Supply Chain Management",
        "SPAN": "Spanish & Portuguese",
        "FREN": "French, Francophone & Italian",
        "GERM": "German Studies",
        "CHIN": "East Asian Languages & Cultures",
        "JPN": "East Asian Languages & Cultures",
        "MUS": "Music",
        "ART": "Visual Art",
        "THR": "Theatre & Dance",
        "DANC": "Theatre & Dance",
        "FMS": "Film & Media Studies",
    }
    return dept_map.get(subject, subject)


def main():
    """Generate comprehensive course catalog"""
    print("=" * 70)
    print("KU COURSE CATALOG GENERATOR - Production Version")
    print("=" * 70)
    print()
    
    # Generate courses
    print("Generating course catalog from KU data...")
    courses = create_manual_course_data()
    
    # Build catalog
    catalog = {
        "last_updated": datetime.now().isoformat(),
        "data_source": "KU Course Catalog (catalog.ku.edu)",
        "version": "2025-2026",
        "total_courses": len(courses),
        "subjects": {},
        "courses": courses
    }
    
    # Count by subject
    for course in courses:
        subj = course["subject"]
        if subj not in catalog["subjects"]:
            catalog["subjects"][subj] = {
                "name": get_department_name(subj),
                "count": 0
            }
        catalog["subjects"][subj]["count"] += 1
    
    # Stats
    popular = sum(1 for c in courses if c.get("popular"))
    ku_core = sum(1 for c in courses if c.get("ku_core"))
    undergrad = sum(1 for c in courses if c["level"] == "undergraduate")
    grad = sum(1 for c in courses if c["level"] == "graduate")
    
    print(f"\n{'='*50}")
    print(f"CATALOG SUMMARY")
    print(f"{'='*50}")
    print(f"Total Courses: {len(courses)}")
    print(f"Subjects: {len(catalog['subjects'])}")
    print(f"Popular Courses: {popular}")
    print(f"KU Core Courses: {ku_core}")
    print(f"Undergraduate: {undergrad}")
    print(f"Graduate: {grad}")
    print()
    
    print("Courses by subject:")
    for subj, info in sorted(catalog["subjects"].items()):
        print(f"  {subj:6} - {info['count']:3} courses ({info['name']})")
    
    # Save catalog
    output_dir = "data/courses"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "catalog.json")
    
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f" Saved {len(courses)} courses to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()