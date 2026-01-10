"""
Embeddings Module - Load data into ChromaDB
============================================
Converts dining, transit, course, building, office, professor,
admission, calendar, and FAQ data into vector embeddings
and stores them in ChromaDB for semantic search.

Usage:
    from app.rag.embeddings import initialize_database
    db = initialize_database()
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Use OpenAI embeddings
EMBEDDING_MODEL = "text-embedding-3-small"


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_hours(hours: Dict) -> str:
    """Format hours dictionary into readable string"""
    if not hours:
        return "Hours not available"
    
    parts = []
    for day, time in hours.items():
        if time and time != "Closed":
            parts.append(f"{day}: {time}")
    
    return "; ".join(parts) if parts else "Hours vary"


def prepare_dining_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert dining data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    locations = data.get("locations", [])
    
    for loc in locations:
        doc = f"""
        Dining Location: {loc['name']}
        Type: {loc['type']}
        Building: {loc['building']}
        Description: {loc.get('description', 'N/A')}
        Hours: {format_hours(loc.get('hours', {}))}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "dining",
            "name": loc['name'],
            "type": loc['type'],
            "building": loc['building'],
            "latitude": loc.get('coordinates', {}).get('latitude', 0),
            "longitude": loc.get('coordinates', {}).get('longitude', 0),
        })
        ids.append(f"dining_{loc['id']}")
    
    return documents, metadatas, ids


def prepare_transit_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert transit data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    routes = data.get("routes", [])
    
    for route in routes:
        doc = f"""
        Bus Route: {route['route_name']}
        Route Number: {route['route_number']}
        Description: {route.get('description', 'N/A')}
        Operating Days: {', '.join(route.get('operates_days', []))}
        Serves KU Campus: {'Yes' if route.get('serves_ku') else 'No'}
        Campus Only: {'Yes' if route.get('campus_only') else 'No'}
        Popular for Students: {'Yes' if route.get('popular_for_students') else 'No'}
        Number of Stops: {len(route.get('stops', []))}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "transit",
            "route_id": route['route_number'],
            "route_name": route['route_name'],
            "serves_ku": route.get('serves_ku', False),
            "campus_only": route.get('campus_only', False),
            "popular_for_students": route.get('popular_for_students', False),
        })
        ids.append(f"transit_route_{route['route_number']}")
    
    return documents, metadatas, ids


def prepare_course_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert course data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    courses = data.get("courses", [])
    
    for course in courses:
        doc = f"""
        Course: {course['course_code']} - {course['title']}
        Department: {course.get('department', 'N/A')}
        Credits: {course['credits']}
        Level: {course['level']}
        Description: {course['description']}
        Prerequisites: {course.get('prerequisites', 'None')}
        KU Core: {course.get('ku_core', 'Not a KU Core course')}
        Popular Course: {'Yes' if course.get('popular') else 'No'}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "course",
            "course_code": course['course_code'],
            "subject": course['subject'],
            "number": course['number'],
            "title": course['title'],
            "credits": course['credits'],
            "level": course['level'],
            "popular": course.get('popular', False),
            "ku_core": course.get('ku_core') or "None",
        })
        ids.append(f"course_{course['subject']}_{course['number']}")
    
    return documents, metadatas, ids


def prepare_building_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert building data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    buildings = data.get("buildings", [])
    
    for building in buildings:
        offices_text = ""
        if building.get('offices'):
            offices_list = [f"{o['name']} (Room {o.get('room', 'N/A')})" for o in building['offices']]
            offices_text = f"Offices: {', '.join(offices_list)}"
        
        doc = f"""
        Building: {building['name']}
        Address: {building.get('address', 'N/A')}
        Phone: {building.get('phone', 'N/A')}
        Departments: {', '.join(building.get('departments', []))}
        Description: {building.get('description', 'N/A')}
        {offices_text}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "building",
            "name": building['name'],
            "address": building.get('address', ''),
        })
        ids.append(f"building_{building['id']}")
    
    return documents, metadatas, ids


def prepare_office_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert office data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    offices = data.get("offices", [])
    
    for office in offices:
        services_text = ', '.join(office.get('services', []))
        
        doc = f"""
        Office: {office['name']}
        Building: {office.get('building', 'N/A')}
        Room: {office.get('room', 'N/A')}
        Address: {office.get('address', 'N/A')}
        Phone: {office.get('phone', 'N/A')}
        Email: {office.get('email', 'N/A')}
        Hours: {office.get('hours', 'N/A')}
        Services: {services_text}
        Description: {office.get('description', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "office",
            "name": office['name'],
            "building": office.get('building', ''),
            "room": office.get('room', ''),
            "phone": office.get('phone', ''),
            "email": office.get('email', ''),
        })
        ids.append(f"office_{office['id']}")
    
    return documents, metadatas, ids


def prepare_professor_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert professor data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    professors = data.get("professors", [])
    
    for prof in professors:
        research_text = ', '.join(prof.get('research_areas', []))
        
        doc = f"""
        Professor: {prof['name']}
        Title: {prof.get('title', 'N/A')}
        Role: {prof.get('role', 'N/A')}
        Department: {prof.get('department', 'N/A')}
        Building: {prof.get('building', 'N/A')}
        Room: {prof.get('room', 'N/A')}
        Phone: {prof.get('phone', 'N/A')}
        Email: {prof.get('email', 'N/A')}
        Research Areas: {research_text if research_text else 'N/A'}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "professor",
            "name": prof['name'],
            "department": prof.get('department', ''),
            "building": prof.get('building', ''),
            "room": prof.get('room', ''),
            "email": prof.get('email', ''),
        })
        ids.append(f"professor_{prof['id']}")
    
    return documents, metadatas, ids


# ============== NEW FUNCTIONS FOR ADMISSIONS, CALENDAR, FAQS ==============

def prepare_admission_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert admissions data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    admissions = data.get("admissions", [])
    
    for admission in admissions:
        # Build requirements text
        reqs_text = ""
        if admission.get('requirements'):
            reqs_text = "; ".join(admission['requirements'])
        elif admission.get('application_requirements'):
            reqs_text = "; ".join(admission['application_requirements'])
        elif admission.get('assured_admission_requirements'):
            reqs_text = "; ".join(admission['assured_admission_requirements'])
        
        # Build deadlines text
        deadlines_text = ""
        if admission.get('deadlines'):
            deadline_parts = []
            for d in admission['deadlines']:
                deadline_parts.append(f"{d.get('name', '')}: {d.get('date', '')}")
            deadlines_text = "; ".join(deadline_parts)
        
        # Build contact text
        contact_text = ""
        if admission.get('contact'):
            c = admission['contact']
            contact_text = f"Contact: {c.get('office', '')} | Phone: {c.get('phone', '')} | Email: {c.get('email', '')}"
        
        doc = f"""
        Admission Type: {admission.get('title', admission.get('type', 'N/A'))}
        Category: {admission.get('category', 'N/A')}
        Description: {admission.get('description', 'N/A')}
        Application Fee: {admission.get('application_fee', 'N/A')}
        Requirements: {reqs_text if reqs_text else 'N/A'}
        Deadlines: {deadlines_text if deadlines_text else 'N/A'}
        {contact_text}
        Website: {admission.get('url', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "admission",
            "type": admission.get('type', ''),
            "category": admission.get('category', ''),
            "title": admission.get('title', ''),
        })
        ids.append(f"admission_{admission.get('id', admission.get('type', 'unknown'))}")
    
    # Add quick facts if available
    if data.get("quick_facts"):
        facts = data["quick_facts"]
        doc = f"""
        KU Admission Quick Facts:
        Acceptance Rate: {facts.get('acceptance_rate', 'N/A')}
        Test Optional: {facts.get('test_optional', 'N/A')}
        Application Fee (Freshman): {facts.get('application_fee_freshman', 'N/A')}
        Application Fee (International): {facts.get('application_fee_international', 'N/A')}
        FAFSA School Code: {facts.get('fafsa_school_code', 'N/A')}
        Scholarship Deadline: {facts.get('scholarship_deadline', 'N/A')}
        Freshman Assured GPA: {facts.get('freshman_assured_gpa', 'N/A')}
        Transfer Assured GPA: {facts.get('transfer_assured_gpa', 'N/A')}
        Transfer Assured Credits: {facts.get('transfer_assured_credits', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "admission",
            "type": "quick_facts",
            "category": "all",
            "title": "Quick Facts",
        })
        ids.append("admission_quick_facts")
    
    return documents, metadatas, ids


def prepare_calendar_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert academic calendar data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    calendar = data.get("academic_calendar", {})
    semesters = calendar.get("semesters", [])
    
    for semester in semesters:
        key_dates = semester.get("key_dates", {})
        
        # Build holidays text
        holidays_text = ""
        holidays = semester.get('holidays_and_breaks') or semester.get('holidays', [])
        if holidays:
            holiday_parts = [f"{h.get('name', '')}: {h.get('date', '')}" for h in holidays]
            holidays_text = "; ".join(holiday_parts)
        
        doc = f"""
        Academic Calendar: {semester.get('name', 'N/A')}
        First Day of Classes: {key_dates.get('first_day_of_classes', 'N/A')}
        Last Day of Classes: {key_dates.get('last_day_of_classes', 'N/A')}
        Finals Start: {key_dates.get('finals_start', 'N/A')}
        Finals End: {key_dates.get('finals_end', 'N/A')}
        Commencement: {key_dates.get('commencement', 'N/A')}
        Grade Submission Deadline: {key_dates.get('grade_submission_deadline', 'N/A')}
        Census Day: {semester.get('census_day', 'N/A')}
        Holidays and Breaks: {holidays_text if holidays_text else 'N/A'}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "calendar",
            "semester": semester.get('name', ''),
            "semester_id": semester.get('id', ''),
            "type": "main",
        })
        ids.append(f"calendar_{semester.get('id', 'unknown')}")
        
        # Create separate document for add/drop deadlines
        if semester.get('add_drop_deadlines'):
            deadline_parts = []
            for d in semester['add_drop_deadlines']:
                deadline_parts.append(f"{d.get('event', '')}: {d.get('date', '')}")
            
            doc = f"""
            Add/Drop Deadlines for {semester.get('name', 'N/A')}:
            {chr(10).join(deadline_parts)}
            Late Enrollment Fee: {semester.get('late_enrollment_fee', 'N/A')}
            """.strip()
            
            documents.append(doc)
            metadatas.append({
                "source": "calendar",
                "semester": semester.get('name', ''),
                "semester_id": semester.get('id', ''),
                "type": "add_drop",
            })
            ids.append(f"calendar_adddrop_{semester.get('id', 'unknown')}")
        
        # Create separate document for refund schedule
        if semester.get('refund_schedule'):
            refund_parts = []
            for r in semester['refund_schedule']:
                period = r.get('period', r.get('refund_percent', ''))
                date = r.get('last_day', r.get('date', ''))
                refund_parts.append(f"{period}: {date}")
            
            doc = f"""
            Refund Schedule for {semester.get('name', 'N/A')}:
            {chr(10).join(refund_parts)}
            """.strip()
            
            documents.append(doc)
            metadatas.append({
                "source": "calendar",
                "semester": semester.get('name', ''),
                "semester_id": semester.get('id', ''),
                "type": "refund",
            })
            ids.append(f"calendar_refund_{semester.get('id', 'unknown')}")
        
        # Create separate document for graduation deadlines
        if semester.get('graduation'):
            grad = semester['graduation']
            doc = f"""
            Graduation Deadlines for {semester.get('name', 'N/A')}:
            Application Available: {grad.get('application_available', 'N/A')}
            Undergraduate Deadline: {grad.get('undergraduate_deadline', 'N/A')}
            Graduate Deadline: {grad.get('graduate_deadline', 'N/A')}
            Course Completion Deadline: {grad.get('course_completion_deadline', 'N/A')}
            Diplomas Available: {grad.get('diplomas_available', 'N/A')}
            """.strip()
            
            documents.append(doc)
            metadatas.append({
                "source": "calendar",
                "semester": semester.get('name', ''),
                "semester_id": semester.get('id', ''),
                "type": "graduation",
            })
            ids.append(f"calendar_graduation_{semester.get('id', 'unknown')}")
    
    return documents, metadatas, ids


def prepare_faq_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert FAQ data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    faqs = data.get("faqs", [])
    
    for faq in faqs:
        # Build steps text
        steps_text = ""
        if faq.get('steps'):
            steps_text = "Steps: " + "; ".join(faq['steps'])
        
        # Build support text
        support_text = ""
        if faq.get('support'):
            s = faq['support']
            support_text = f"Support: Phone: {s.get('phone', 'N/A')} | Email: {s.get('email', 'N/A')}"
        
        doc = f"""
        FAQ: {faq.get('question', 'N/A')}
        Category: {faq.get('category', 'N/A')}
        Answer: {faq.get('answer', 'N/A')}
        {steps_text}
        {support_text}
        More Info: {faq.get('url', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "faq",
            "category": faq.get('category', ''),
            "question": faq.get('question', ''),
        })
        ids.append(f"faq_{faq.get('id', 'unknown')}")
    
    # Add contact info
    if data.get("contact_info"):
        for name, info in data["contact_info"].items():
            doc = f"""
            Contact: {info.get('name', name)}
            Phone: {info.get('phone', 'N/A')}
            Email: {info.get('email', 'N/A')}
            Location: {info.get('location', 'N/A')}
            Hours: {info.get('hours', 'N/A')}
            """.strip()
            
            documents.append(doc)
            metadatas.append({
                "source": "faq",
                "category": "contact",
                "question": f"Contact info for {info.get('name', name)}",
            })
            ids.append(f"faq_contact_{name}")
    
    # Add key URLs
    if data.get("key_urls"):
        url_parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in data["key_urls"].items()]
        doc = f"""
        Important KU Websites and URLs:
        {chr(10).join(url_parts)}
        """.strip()
        
        documents.append(doc)
        metadatas.append({
            "source": "faq",
            "category": "urls",
            "question": "Important KU websites",
        })
        ids.append("faq_key_urls")
    
    return documents, metadatas, ids


def prepare_tuition_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert tuition and fees data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    tuition_data = data.get("tuition_and_fees", {})
    
    # Base tuition rates document
    base_rates = tuition_data.get("base_tuition_rates", {}).get("lawrence_edwards_campus", {})
    if base_rates:
        undergrad = base_rates.get("undergraduate", {})
        grad = base_rates.get("graduate", {})
        
        doc = f"""
        KU Tuition Rates (2025-2026):
        
        UNDERGRADUATE:
        Kansas Resident: {undergrad.get('resident', {}).get('per_credit_hour', 'N/A')} per credit hour ({undergrad.get('resident', {}).get('estimated_annual_30_hours', 'N/A')} annual for 30 hours)
        Non-Resident: {undergrad.get('non_resident', {}).get('per_credit_hour', 'N/A')} per credit hour ({undergrad.get('non_resident', {}).get('estimated_annual_30_hours', 'N/A')} annual for 30 hours)
        
        GRADUATE:
        Kansas Resident: {grad.get('resident', {}).get('per_credit_hour', 'N/A')} per credit hour
        Non-Resident: {grad.get('non_resident', {}).get('per_credit_hour', 'N/A')} per credit hour
        
        Note: All rates include $10.00 technology fee per credit hour
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "base_rates", "category": "tuition"})
        ids.append("tuition_base_rates")
    
    # Mandatory fees document
    fees = tuition_data.get("mandatory_fees", {})
    if fees:
        student_fee = fees.get("student_fee", {})
        wellness_fee = fees.get("wellness_fee", {})
        
        doc = f"""
        KU Mandatory Fees:
        
        STUDENT FEE (Undergraduate Fall/Spring):
        0-11.99 hours: {student_fee.get('undergraduate', {}).get('fall_spring', {}).get('0_to_11.99_hours', 'N/A')}
        12+ hours: {student_fee.get('undergraduate', {}).get('fall_spring', {}).get('12_plus_hours', 'N/A')}
        
        WELLNESS FEE (All Students Fall/Spring):
        0-2.99 hours: {wellness_fee.get('all_students', {}).get('fall_spring', {}).get('0_to_2.99_hours', 'N/A')}
        3+ hours: {wellness_fee.get('all_students', {}).get('fall_spring', {}).get('3_plus_hours', 'N/A')}
        
        INFRASTRUCTURE FEE: {fees.get('infrastructure_fee', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "mandatory_fees", "category": "fees"})
        ids.append("tuition_mandatory_fees")
    
    # College/School fees document
    college_fees = tuition_data.get("college_school_fees_per_credit_hour", {})
    if college_fees:
        fees_text = "\n        ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in college_fees.items()])
        doc = f"""
        KU College/School Course Fees (per credit hour, in addition to tuition):
        {fees_text}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "college_fees", "category": "fees"})
        ids.append("tuition_college_fees")
    
    # Payment information document
    payment = tuition_data.get("payment_information", {})
    if payment:
        billing = payment.get("billing_cycle", {})
        late_fees = payment.get("late_fees", {})
        plan = payment.get("payment_plan", {})
        
        doc = f"""
        KU Tuition Payment Information:
        
        BILLING CYCLE:
        {billing.get('description', 'N/A')}
        Fall Initial Bill: {billing.get('fall_initial_bill', 'N/A')}
        Spring Initial Bill: {billing.get('spring_initial_bill', 'N/A')}
        Summer Initial Bill: {billing.get('summer_initial_bill', 'N/A')}
        
        LATE FEES:
        First Late Fee: {late_fees.get('first_late_fee', 'N/A')}
        Second Late Fee: {late_fees.get('second_late_fee', 'N/A')}
        Summer Late Fee: {late_fees.get('summer_late_fee', 'N/A')}
        Default Fee: {late_fees.get('default_fee', 'N/A')}
        
        PAYMENT PLAN (Nelnet):
        Enrollment Fee: {plan.get('enrollment_fee', 'N/A')}
        Payment Date: {plan.get('payment_date', 'N/A')}
        How to Enroll: {plan.get('how_to_enroll', 'N/A')}
        
        PAY ONLINE: Enroll & Pay (sa.ku.edu)
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "payment_info", "category": "payment"})
        ids.append("tuition_payment_info")
    
    # Cost of attendance document
    coa = tuition_data.get("estimated_cost_of_attendance", {})
    if coa:
        resident = coa.get("undergraduate_resident", {})
        nonres = coa.get("undergraduate_non_resident", {})
        
        doc = f"""
        KU Estimated Cost of Attendance (2024-2025):
        
        KANSAS RESIDENT (On Campus):
        Tuition & Fees: {resident.get('tuition_fees', 'N/A')}
        Room & Board: {resident.get('room_board', 'N/A')}
        Books & Supplies: {resident.get('books_supplies', 'N/A')}
        Transportation: {resident.get('transportation', 'N/A')}
        Personal Expenses: {resident.get('personal_expenses', 'N/A')}
        TOTAL: {resident.get('total_on_campus', 'N/A')}
        
        NON-RESIDENT (On Campus):
        Tuition & Fees: {nonres.get('tuition_fees', 'N/A')}
        Room & Board: {nonres.get('room_board', 'N/A')}
        Books & Supplies: {nonres.get('books_supplies', 'N/A')}
        Transportation: {nonres.get('transportation', 'N/A')}
        Personal Expenses: {nonres.get('personal_expenses', 'N/A')}
        TOTAL: {nonres.get('total_on_campus', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "cost_of_attendance", "category": "cost"})
        ids.append("tuition_cost_of_attendance")
    
    # Contact information
    contact = tuition_data.get("contact", {})
    if contact:
        doc = f"""
        Student Accounts & Receivables Contact:
        Office: {contact.get('office', 'N/A')}
        Address: {contact.get('address', 'N/A')}
        Phone: {contact.get('phone', 'N/A')}
        Email: {contact.get('email', 'N/A')}
        Website: {contact.get('website', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "contact", "category": "contact"})
        ids.append("tuition_contact")
    
    # Add tuition FAQs
    for faq in data.get("tuition_faqs", []):
        doc = f"""
        Tuition FAQ: {faq.get('question', 'N/A')}
        Answer: {faq.get('answer', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "tuition", "type": "faq", "category": faq.get('category', 'general')})
        ids.append(f"tuition_{faq.get('id', 'unknown')}")
    
    return documents, metadatas, ids


def prepare_financial_aid_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert financial aid data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    finaid = data.get("financial_aid", {})
    
    # FAFSA information
    fafsa = finaid.get("fafsa", {})
    if fafsa:
        doc = f"""
        FAFSA Information for KU:
        Description: {fafsa.get('description', 'N/A')}
        Website: {fafsa.get('website', 'N/A')}
        KU School Code: {fafsa.get('ku_school_code', 'N/A')}
        Priority Deadline: {fafsa.get('priority_deadline', 'N/A')}
        FAFSA Opens: {fafsa.get('opens', 'N/A')}
        Important Notes: {'; '.join(fafsa.get('important_notes', []))}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "financial_aid", "type": "fafsa", "category": "fafsa"})
        ids.append("finaid_fafsa")
    
    # Grants information
    grants = finaid.get("grants", {})
    if grants:
        for grant in grants.get("types", []):
            doc = f"""
            Grant: {grant.get('name', 'N/A')}
            Type: {grant.get('type', 'N/A')}
            Amount: {grant.get('amount', 'Varies')}
            Deadline: {grant.get('deadline', 'N/A')}
            Eligibility: {grant.get('eligibility', 'N/A')}
            """.strip()
            
            if grant.get('renewal_requirements'):
                doc += f"\n        Renewal Requirements: {'; '.join(grant.get('renewal_requirements', []))}"
            
            documents.append(doc)
            metadatas.append({"source": "financial_aid", "type": "grant", "name": grant.get('name', '')})
            ids.append(f"finaid_grant_{grant.get('name', 'unknown').lower().replace(' ', '_')}")
    
    # Scholarships information
    scholarships = finaid.get("scholarships", {})
    if scholarships:
        # Freshman scholarships
        freshman = scholarships.get("freshman_scholarships", {})
        ks_awards = freshman.get("kansas_resident_awards", {})
        oos_awards = freshman.get("out_of_state_awards", {})
        
        doc = f"""
        KU Freshman Scholarships:
        Deadline: {freshman.get('deadline', 'N/A')}
        Based On: {freshman.get('based_on', 'N/A')}
        
        KANSAS RESIDENT AWARDS:
        4.0 GPA: {ks_awards.get('3.9_4.0_gpa', 'N/A')}/year
        3.75-3.89 GPA: {ks_awards.get('3.75_3.89_gpa', 'N/A')}/year
        3.5-3.74 GPA: {ks_awards.get('3.5_3.74_gpa', 'N/A')}/year
        3.25-3.49 GPA: {ks_awards.get('3.25_3.49_gpa', 'N/A')}/year
        Maximum 4-Year Total: {ks_awards.get('max_4_year_total', 'N/A')}
        
        OUT-OF-STATE AWARDS:
        4.0 GPA: {oos_awards.get('4.0_gpa', 'N/A')}/year
        3.9-3.99 GPA: {oos_awards.get('3.9_3.99_gpa', 'N/A')}/year
        3.75-3.89 GPA: {oos_awards.get('3.75_3.89_gpa', 'N/A')}/year
        3.5-3.74 GPA: {oos_awards.get('3.5_3.74_gpa', 'N/A')}/year
        3.25-3.49 GPA: {oos_awards.get('3.25_3.49_gpa', 'N/A')}/year
        Maximum 4-Year Total: {oos_awards.get('max_4_year_total', 'N/A')}
        
        National Merit Finalist Bonus: {freshman.get('national_merit_finalist', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "financial_aid", "type": "scholarship", "category": "freshman"})
        ids.append("finaid_freshman_scholarships")
        
        # Scholarship renewal requirements
        renewal = scholarships.get("renewal_requirements", {})
        if renewal:
            doc = f"""
            Scholarship Renewal Requirements:
            GPA Required: {renewal.get('gpa', 'N/A')}
            Enrollment: {renewal.get('enrollment', 'N/A')}
            Freshman Scholarships Expire: {renewal.get('freshman_expires', 'N/A')}
            Transfer Scholarships Expire: {renewal.get('transfer_expires', 'N/A')}
            Reinstatement: {renewal.get('reinstatement', 'N/A')}
            """.strip()
            
            documents.append(doc)
            metadatas.append({"source": "financial_aid", "type": "scholarship", "category": "renewal"})
            ids.append("finaid_scholarship_renewal")
    
    # Work-study information
    workstudy = finaid.get("work_study", {})
    if workstudy:
        fws = workstudy.get("federal_work_study", {})
        dates = fws.get("important_dates_2025_26", {})
        
        doc = f"""
        Federal Work-Study at KU:
        Description: {workstudy.get('description', 'N/A')}
        Type: {fws.get('type', 'N/A')}
        Eligibility: {fws.get('eligibility', 'N/A')}
        Hours: {fws.get('hours', 'N/A')}
        Pay: {fws.get('pay', 'N/A')}
        How It Works: {fws.get('how_it_works', 'N/A')}
        
        Important Dates 2025-26:
        Last Day Summer 2025 Funds: {dates.get('last_day_summer_2025_funds', 'N/A')}
        First Day Fall Funds: {dates.get('first_day_fall_funds', 'N/A')}
        Last Day Fall Funds: {dates.get('last_day_fall_funds', 'N/A')}
        First Day Spring Funds: {dates.get('first_day_spring_funds', 'N/A')}
        Last Day Spring Funds: {dates.get('last_day_spring_funds', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "financial_aid", "type": "work_study", "category": "employment"})
        ids.append("finaid_work_study")
    
    # Loans information
    loans = finaid.get("loans", {})
    if loans:
        loan_types = loans.get("types", [])
        loan_text = "\n        ".join([f"{l.get('name', 'N/A')}: {l.get('type', '')} - {l.get('interest', l.get('note', ''))}" for l in loan_types])
        
        doc = f"""
        Student Loans at KU:
        {loan_text}
        
        Important Notes:
        {'; '.join(loans.get('important_notes', []))}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "financial_aid", "type": "loans", "category": "loans"})
        ids.append("finaid_loans")
    
    # Contact information
    contact = finaid.get("contact", {})
    if contact:
        doc = f"""
        Financial Aid & Scholarships Contact:
        Office: {contact.get('office', 'N/A')}
        Address: {contact.get('address', 'N/A')}
        Phone: {contact.get('phone', 'N/A')}
        Email: {contact.get('email', 'N/A')}
        Website: {contact.get('website', 'N/A')}
        Hours: {contact.get('hours', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "financial_aid", "type": "contact", "category": "contact"})
        ids.append("finaid_contact")
    
    # Add financial aid FAQs
    for faq in data.get("financial_aid_faqs", []):
        doc = f"""
        Financial Aid FAQ: {faq.get('question', 'N/A')}
        Answer: {faq.get('answer', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "financial_aid", "type": "faq", "category": faq.get('category', 'general')})
        ids.append(f"finaid_{faq.get('id', 'unknown')}")
    
    return documents, metadatas, ids


def prepare_housing_documents(data: Dict) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert housing data into documents for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    housing = data.get("housing", {})
    
    # General housing info
    general = housing.get("general_info", {})
    if general:
        doc = f"""
        KU Housing General Information:
        {general.get('description', 'N/A')}
        Application Fee: {general.get('application_fee', 'N/A')}
        All Rates Include: {general.get('all_rates_include', 'N/A')}
        Dining Plan Required: {general.get('dining_plan_required', 'N/A')}
        Financial Aid: {general.get('financial_aid_applies', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "housing", "type": "general", "category": "info"})
        ids.append("housing_general")
    
    # Residence halls
    res_halls = housing.get("residence_halls", {})
    if res_halls:
        for hall in res_halls.get("locations", []):
            rates = hall.get("rates_2026_27", {})
            rates_text = "\n            ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in rates.items()])
            
            doc = f"""
            Residence Hall: {hall.get('name', 'N/A')}
            Type: {hall.get('type', 'residence_hall')}
            Area: {hall.get('area', 'Main Campus')}
            Room Types: {', '.join(hall.get('room_types', []))}
            Bath: {hall.get('bath', 'N/A')}
            
            2026-2027 Rates:
            {rates_text}
            """.strip()
            
            documents.append(doc)
            metadatas.append({"source": "housing", "type": "residence_hall", "name": hall.get('name', '')})
            ids.append(f"housing_reshall_{hall.get('name', 'unknown').lower().replace(' ', '_')}")
    
    # Scholarship halls
    schol_halls = housing.get("scholarship_halls", {})
    if schol_halls:
        halls_text = []
        for hall in schol_halls.get("halls", []):
            halls_text.append(f"{hall.get('name', 'N/A')}: {hall.get('rate_2026_27', 'N/A')} (Dining: {hall.get('dining_cost', 'N/A')})")
        
        doc = f"""
        Scholarship Halls at KU:
        Description: {schol_halls.get('description', 'N/A')}
        Application Deadline: {schol_halls.get('application_deadline', 'N/A')}
        Cheapest Option: {schol_halls.get('cheapest_option', 'N/A')}
        
        Halls and Rates:
        {chr(10).join(halls_text)}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "housing", "type": "scholarship_hall", "category": "schol_halls"})
        ids.append("housing_scholarship_halls")
    
    # Apartments
    apartments = housing.get("apartments", {})
    if apartments:
        for apt in apartments.get("locations", []):
            rates = apt.get("rates_2026_27", {})
            rates_text = "\n            ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in rates.items()])
            
            doc = f"""
            Apartment: {apt.get('name', 'N/A')}
            Note: {apt.get('note', 'Upper-class, transfer, non-traditional students')}
            
            2026-2027 Rates:
            {rates_text}
            """.strip()
            
            documents.append(doc)
            metadatas.append({"source": "housing", "type": "apartment", "name": apt.get('name', '')})
            ids.append(f"housing_apt_{apt.get('name', 'unknown').lower().replace(' ', '_')}")
    
    # Dining plans
    dining = housing.get("dining_plans", {})
    if dining:
        plans_text = []
        for plan in dining.get("plans", []):
            plans_text.append(f"{plan.get('name', 'N/A')}: {plan.get('cost_per_semester', plan.get('cost', 'N/A'))}/semester, {plan.get('swipes', 'N/A')} swipes, ${plan.get('dining_dollars_per_semester', plan.get('dining_dollars', 'N/A'))} dining dollars")
        
        doc = f"""
        KU Dining Plans (2025-2026):
        Required For: {dining.get('required_for', 'N/A')}
        
        Plans:
        {chr(10).join(plans_text)}
        
        AYCTE Dining Halls: {', '.join(dining.get('dining_halls_aycte', []))}
        Retail Locations: {', '.join(dining.get('retail_locations', []))}
        
        Important: {dining.get('dining_dollars_note', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "housing", "type": "dining_plans", "category": "dining"})
        ids.append("housing_dining_plans")
    
    # Application process
    app_process = housing.get("application_process", {})
    if app_process:
        first_year = app_process.get("first_year_students", {})
        
        doc = f"""
        Housing Application Process - First Year Students:
        Application Opens: {first_year.get('application_opens', 'N/A')}
        Priority Deadline: {first_year.get('priority_deadline', 'N/A')}
        Enrollment Deposit Required: {first_year.get('enrollment_deposit_required', 'N/A')}
        Room Selection: {first_year.get('room_selection', 'N/A')}
        
        How to Apply:
        {chr(10).join(first_year.get('how_to_apply', []))}
        
        Scholarship Halls Deadline: {app_process.get('scholarship_halls', {}).get('application_deadline', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "housing", "type": "application", "category": "deadlines"})
        ids.append("housing_application_process")
    
    # Contact information
    contact = housing.get("contact", {})
    if contact:
        doc = f"""
        Housing & Residence Life Contact:
        Office: {contact.get('office', 'N/A')}
        Address: {contact.get('address', 'N/A')}
        Phone: {contact.get('phone', 'N/A')}
        Email: {contact.get('email', 'N/A')}
        Website: {contact.get('website', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "housing", "type": "contact", "category": "contact"})
        ids.append("housing_contact")
    
    # Add housing FAQs
    for faq in data.get("housing_faqs", []):
        doc = f"""
        Housing FAQ: {faq.get('question', 'N/A')}
        Answer: {faq.get('answer', 'N/A')}
        """.strip()
        
        documents.append(doc)
        metadatas.append({"source": "housing", "type": "faq", "category": faq.get('category', 'general')})
        ids.append(f"housing_{faq.get('id', 'unknown')}")
    
    return documents, metadatas, ids


def prepare_library_documents(library_data: dict) -> Tuple[list, list, list]:
    """
    Convert libraries JSON into documents for embedding.
    Returns (documents, metadatas, ids) tuples.
    """
    documents = []
    metadatas = []
    ids = []
    
    # Overview document
    overview = library_data.get("overview", {})
    overview_text = f"""KU Libraries Overview:
        {overview.get('description', '')}
        KU Libraries has {overview.get('total_items', '')} in {overview.get('campus_locations', '')} campus locations.
        The libraries receive {overview.get('annual_visits', '')} visits annually.
        Main phone: {overview.get('main_phone', '')}
        Email: {overview.get('main_email', '')}
        Website: {overview.get('website', '')}
        Library catalog: {overview.get('catalog_url', '')}
        Ask a Librarian: {overview.get('ask_librarian_url', '')}
        Reserve study rooms: {overview.get('reserve_rooms_url', '')}"""
    
    documents.append(overview_text)
    metadatas.append({"source": "libraries", "type": "overview"})
    ids.append("lib_overview")
    
    # Individual library documents
    for lib in library_data.get("libraries", []):
        lib_name = lib.get("name", "")
        
        # Main library info
        lib_text = f"""{lib.get('full_name', lib_name)}:
            {lib.get('description', '')}
            Named for: {lib.get('named_for', 'N/A')}
            Address: {lib.get('address', '')}
            Phone: {lib.get('phone', '')}"""
        
        # Add hours if available
        hours = lib.get("hours", {})
        if hours:
            lib_text += "\nHours: "
            if isinstance(hours, dict):
                for period, times in hours.items():
                    if isinstance(times, dict):
                        for day, time in times.items():
                            lib_text += f"\n  {day.replace('_', ' ').title()}: {time}"
                    elif period == "note":
                        lib_text += f"\n  Note: {times}"
        
        # Add collections
        collections = lib.get("collections", [])
        if collections:
            if isinstance(collections, list):
                # Handle list of dicts or list of strings
                collection_names = []
                for c in collections:
                    if isinstance(c, dict):
                        collection_names.append(c.get('name', str(c)))
                    else:
                        collection_names.append(str(c))
                lib_text += f"\nCollections: {', '.join(collection_names)}"
            else:
                lib_text += f"\nCollections: {collections}"
        
        # Add services
        services = lib.get("services_located_here", [])
        if services:
            lib_text += f"\nServices: {', '.join(services)}"
        
        # Add equipment checkout
        equipment = lib.get("equipment_checkout", [])
        if equipment:
            lib_text += f"\nEquipment available for checkout: {', '.join(equipment)}"
        
        documents.append(lib_text)
        metadatas.append({"source": "libraries", "type": "library", "name": lib_name})
        ids.append(f"lib_{lib_name.lower().replace(' ', '_').replace('&', 'and')}")
        
        # Study rooms document if available
        study_rooms = lib.get("study_rooms", {})
        if study_rooms and study_rooms.get("available"):
            room_text = f"""Study Rooms at {lib_name}:
                Reservation URL: {study_rooms.get('reservation_url', '')}
                Maximum hours per day: {study_rooms.get('max_hours_per_day', 2)}"""
            if study_rooms.get("locations"):
                room_text += f"\nLocations: {study_rooms.get('locations')}"
            if study_rooms.get("group_size_required"):
                room_text += f"\nGroup size required: {study_rooms.get('group_size_required')}"
            if study_rooms.get("advance_booking"):
                room_text += f"\nAdvance booking: {study_rooms.get('advance_booking')}"
            
            documents.append(room_text)
            metadatas.append({"source": "libraries", "type": "study_rooms", "library": lib_name})
            ids.append(f"lib_{lib_name.lower().replace(' ', '_')}_rooms")
        
        # Floor information if available
        floors = lib.get("floors", [])
        if floors:
            for floor_info in floors:
                floor_num = floor_info.get("floor", "")
                features = floor_info.get("features", [])
                if features:
                    floor_text = f"""{lib_name} Floor {floor_num}:
                        Features: {', '.join(features)}"""
                    documents.append(floor_text)
                    metadatas.append({"source": "libraries", "type": "floor", "library": lib_name, "floor": floor_num})
                    ids.append(f"lib_{lib_name.lower().replace(' ', '_')}_floor_{floor_num}")
    
    # Special services documents
    special_services = library_data.get("special_services", {})
    
    # Makerspace
    makerspace = special_services.get("makerspace", {})
    if makerspace:
        maker_text = f"""KU Libraries Makerspace:
        Location: {makerspace.get('location', '')}
        Hours: {makerspace.get('hours', '')}
        {makerspace.get('description', '')}

        Services available:"""
        for service in makerspace.get("services", []):
            maker_text += f"\n- {service.get('name', '')}: {', '.join(service.get('equipment', [])) if service.get('equipment') else ''}"
        maker_text += f"""

        To request 3D printing: {makerspace.get('services', [{}])[0].get('request_form', '')}
        Schedule a consultation: {makerspace.get('consultation_url', '')}"""
        
        documents.append(maker_text)
        metadatas.append({"source": "libraries", "type": "service", "name": "makerspace"})
        ids.append("lib_makerspace")
    
    # Studio K
    studio_k = special_services.get("studio_k", {})
    if studio_k:
        studio_text = f"""Studio K - Video Recording Studio:
        Location: {studio_k.get('location', '')}
        {studio_k.get('description', '')}
        Features: {', '.join(studio_k.get('features', []))}
        Reservation: {studio_k.get('reservation_url', '')}"""
        
        documents.append(studio_text)
        metadatas.append({"source": "libraries", "type": "service", "name": "studio_k"})
        ids.append("lib_studio_k")
    
    # GIS & Data Lab
    gis_lab = special_services.get("gis_data_lab", {})
    if gis_lab:
        gis_text = f"""GIS & Data Lab:
        Location: {gis_lab.get('location', '')}
        Phone: {gis_lab.get('phone', '')}
        Hours: {gis_lab.get('hours', '')}
        {gis_lab.get('description', '')}
        Services: {', '.join(gis_lab.get('services', []))}"""
        
        documents.append(gis_text)
        metadatas.append({"source": "libraries", "type": "service", "name": "gis_lab"})
        ids.append("lib_gis_lab")
    
    # Map Collection
    map_collection = special_services.get("tr_smith_map_collection", {})
    if map_collection:
        map_text = f"""T.R. Smith Map Collection:
        Location: {map_collection.get('location', '')}
        Phone: {map_collection.get('phone', '')}
        Hours: {map_collection.get('hours', '')}
        {map_collection.get('description', '')}
        Holdings: {map_collection.get('holdings', {}).get('sheet_maps', '')} sheet maps, {map_collection.get('holdings', {}).get('aerial_photographs', '')} aerial photographs
        Services: {', '.join(map_collection.get('services', []))}"""
        
        documents.append(map_text)
        metadatas.append({"source": "libraries", "type": "service", "name": "map_collection"})
        ids.append("lib_map_collection")
    
    # International Collections
    intl = special_services.get("international_collections", {})
    if intl:
        intl_text = f"""International Collections:
        Location: {intl.get('location', '')}
        {intl.get('description', '')}
        Regional specializations: {', '.join(intl.get('regional_specializations', []))}
        Website: {intl.get('website', '')}"""
        
        documents.append(intl_text)
        metadatas.append({"source": "libraries", "type": "service", "name": "international_collections"})
        ids.append("lib_international_collections")
    
    # Printing and scanning
    printing = library_data.get("printing_scanning", {})
    if printing:
        print_text = f"""Library Printing and Scanning:
        Black & white printing: {printing.get('costs', {}).get('black_and_white', '')}
        Color printing: {printing.get('costs', {}).get('color', '')}
        Scanning: {printing.get('costs', {}).get('scanning', '')}

        Free printing for students:
        - Fall/Spring: {printing.get('free_printing', {}).get('fall_spring', '')}
        - Summer: {printing.get('free_printing', {}).get('summer', '')}
        Provided by: {printing.get('free_printing', {}).get('provided_by', '')}

        Payment method: {printing.get('payment_method', '')}
        Locations: {', '.join(printing.get('locations', []))}

        How to print:
        {chr(10).join(['- ' + step for step in printing.get('how_to_print', [])])}

        Visitor printing:
        - B&W: {printing.get('visitor_printing', {}).get('cost_bw', '')}
        - Color: {printing.get('visitor_printing', {}).get('cost_color', '')}
        - Purchase at: {', '.join(printing.get('visitor_printing', {}).get('purchase_locations', []))}"""
        
        documents.append(print_text)
        metadatas.append({"source": "libraries", "type": "printing"})
        ids.append("lib_printing")
    
    # Borrowing policies
    borrowing = library_data.get("borrowing", {})
    if borrowing:
        borrow_text = f"""Library Borrowing Policies:
        Loan periods:
        - Faculty/Staff/Graduate students: {borrowing.get('loan_periods', {}).get('faculty_staff_grad', '')}
        - Undergraduates: {borrowing.get('loan_periods', {}).get('undergrad', '')}
        - DVDs/Videos: {borrowing.get('loan_periods', {}).get('dvds_videos', '')}
        - 4-hour laptops: {borrowing.get('loan_periods', {}).get('laptops_4hr', '')}
        - 2-week laptops: {borrowing.get('loan_periods', {}).get('laptops_2wk', '')}
        - Course reserves: {borrowing.get('loan_periods', {}).get('course_reserves', '')}

        Renewals:
        - Online renewals: {borrowing.get('renewals', {}).get('online_renewals', '')}
        - URL: {borrowing.get('renewals', {}).get('url', '')}

        Fines:
        - 4-hour laptops: {borrowing.get('fines', {}).get('4hr_laptops', '')}
        - 1-week laptops: {borrowing.get('fines', {}).get('1wk_laptops', '')}
        - Accessories: {borrowing.get('fines', {}).get('accessories', '')}
        - Calculators: {borrowing.get('fines', {}).get('calculators', '')}
        - Borrowing blocked at: {borrowing.get('fines', {}).get('blocked_at', '')}

        Interlibrary Loan:
        - Eligibility: {borrowing.get('interlibrary_loan', {}).get('eligibility', '')}
        - Cost: {borrowing.get('interlibrary_loan', {}).get('cost', '')}
        - Website: {borrowing.get('interlibrary_loan', {}).get('website', '')}"""
        
        documents.append(borrow_text)
        metadatas.append({"source": "libraries", "type": "borrowing"})
        ids.append("lib_borrowing")
    
    # Equipment checkout
    equipment = library_data.get("equipment_checkout", {})
    if equipment:
        equip_text = f"""Library Equipment Checkout:
Available items: {', '.join(equipment.get('available_items', []))}
Locations: {', '.join(equipment.get('locations', []))}
Requirements: {equipment.get('requirements', '')}"""
        
        documents.append(equip_text)
        metadatas.append({"source": "libraries", "type": "equipment"})
        ids.append("lib_equipment")
    
    # Study rooms general
    study_rooms = library_data.get("study_rooms", {})
    if study_rooms:
        rooms_text = f"""Library Study Rooms:
Reservation system: {study_rooms.get('reservation_system', '')}

Policies:
- Max advance booking: {study_rooms.get('policies', {}).get('max_advance_booking', '')}
- Max hours per day: {study_rooms.get('policies', {}).get('max_hours_per_day', '')}
- Group size: {study_rooms.get('policies', {}).get('group_size', '')}
- Grace period: {study_rooms.get('policies', {}).get('grace_period', '')}

Reservation URLs by library:
- Watson: {study_rooms.get('locations', {}).get('watson', '')}
- Anschutz: {study_rooms.get('locations', {}).get('anschutz', '')}
- Art & Architecture: {study_rooms.get('locations', {}).get('art_architecture', '')}
- Spahr/LEEP2: {study_rooms.get('locations', {}).get('spahr_leep2', '')}"""
        
        documents.append(rooms_text)
        metadatas.append({"source": "libraries", "type": "study_rooms_general"})
        ids.append("lib_study_rooms_general")
    
    # Ask a Librarian
    ask = library_data.get("ask_a_librarian", {})
    if ask:
        ask_text = f"""Ask a Librarian - Research Help:
{ask.get('description', '')}
Methods: {', '.join(ask.get('methods', []))}
Website: {ask.get('website', '')}
Response time: {ask.get('response_time', '')}
Services: {', '.join(ask.get('services', []))}"""
        
        documents.append(ask_text)
        metadatas.append({"source": "libraries", "type": "ask_librarian"})
        ids.append("lib_ask_librarian")
    
    # FAQs
    for i, faq in enumerate(library_data.get("faqs", [])):
        faq_text = f"Library FAQ: {faq.get('question', '')}\nAnswer: {faq.get('answer', '')}"
        documents.append(faq_text)
        metadatas.append({"source": "libraries", "type": "faq", "question": faq.get('question', '')})
        ids.append(f"lib_faq_{i+1}")
    
    # Contact information
    contacts = library_data.get("contact_information", {})
    if contacts:
        contact_text = "Library Contact Information:\n"
        
        main = contacts.get("main", {})
        if main:
            contact_text += f"\nMain Contact ({main.get('name', '')}):\nPhone: {main.get('phone', '')}\nEmail: {main.get('email', '')}\n"
        
        by_lib = contacts.get("by_library", {})
        if by_lib:
            contact_text += "\nBy Library:"
            for lib_name, info in by_lib.items():
                contact_text += f"\n  {lib_name.replace('_', ' ').title()}:"
                for key, value in info.items():
                    contact_text += f"\n    {key.replace('_', ' ').title()}: {value}"
        
        by_service = contacts.get("by_service", {})
        if by_service:
            contact_text += "\n\nBy Service:"
            for service_name, info in by_service.items():
                contact_text += f"\n  {service_name.replace('_', ' ').title()}:"
                for key, value in info.items():
                    contact_text += f"\n    {key.title()}: {value}"
        
        documents.append(contact_text)
        metadatas.append({"source": "libraries", "type": "contacts"})
        ids.append("lib_contacts")
    
    return documents, metadatas, ids


def prepare_recreation_documents(recreation_data: dict) -> Tuple[List[str], List[dict], List[str]]:
    """
    Convert recreation JSON data into documents for embedding.
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    doc_counter = 0
    
    # 1. Overview document
    overview = recreation_data.get("overview", {})
    if overview:
        doc = f"""KU Recreation Services Overview

{overview.get('description', '')}

Mission: {overview.get('mission', '')}

Main Facility: {overview.get('main_facility', '')} ({overview.get('abbreviation', 'ASRFC')})
Address: {overview.get('address', '')}
Phone: {overview.get('phone', '')}
Email: {overview.get('email', '')}
Website: {overview.get('website', '')}

The Ambler Student Recreation Fitness Center is {overview.get('total_size', '')} and opened in {overview.get('opened', '')}. It is named after {overview.get('named_after', '')}."""
        
        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "overview"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 2. Ambler SRFC facility document
    facilities = recreation_data.get("facilities", {})
    ambler = facilities.get("ambler_srfc", {})
    if ambler:
        features = ambler.get("features", {})
        
        doc = f"""Ambler Student Recreation Fitness Center (ASRFC)

Address: {ambler.get('address', '')}
Phone: {ambler.get('phone', '')}
Size: {ambler.get('size', '')}

Features:
- Cardio and Weights: {features.get('cardio_and_weights', {}).get('description', '')}
- Basketball/Volleyball Courts: {features.get('courts', {}).get('basketball_volleyball', '')}
- Racquetball Courts: {features.get('courts', {}).get('racquetball', '')}
- Indoor Track: Suspended walking/jogging track elevated over courts
- Climbing Wall: The Chalk Rock, {features.get('climbing_wall', {}).get('height', '42 feet')} tall
- Studios: Aerobics studio, cycle studio, martial arts studio, functional fitness studio
- Other: Table tennis, Teqball table, lawn games checkout, locker rooms

Track Information:
- Inside lane: {features.get('track', {}).get('length', {}).get('inside_lane', '4.75 laps = 1 mile')}
- Middle lane: {features.get('track', {}).get('length', {}).get('middle_lane', '4.5 laps = 1 mile')}
- Outside lane: {features.get('track', {}).get('length', {}).get('outside_lane', '4.25 laps = 1 mile')}
- Direction alternates by day: Counter-clockwise on Mon/Wed/Fri/Sun, Clockwise on Tue/Thu/Sat
- Walkers use inside lane, joggers/runners use outer lanes"""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "facility", "name": "Ambler SRFC"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 3. Hours document
    hours = ambler.get("hours", {})
    if hours:
        spring = hours.get("spring_2025", {})
        spring_break = hours.get("spring_break", {})
        
        doc = f"""Ambler Recreation Center Hours

Spring 2025 Hours ({spring.get('dates', 'January 17 - May 16, 2025')}):
- Monday-Thursday: {spring.get('monday_thursday', '6:00 AM - 10:30 PM')}
- Friday: {spring.get('friday', '6:00 AM - 9:00 PM')}
- Saturday: {spring.get('saturday', '9:00 AM - 5:00 PM')}
- Sunday: {spring.get('sunday', '1:00 PM - 10:30 PM')}

Spring Break Hours ({spring_break.get('dates', 'March 15-22, 2025')}):
- Saturday-Sunday: {spring_break.get('saturday_sunday', '10:00 AM - 3:00 PM')}
- Monday-Friday: {spring_break.get('monday_friday', '7:00 AM - 7:00 PM')}

Admin Office Hours:
- Monday-Friday: {hours.get('admin_office', {}).get('monday_friday', '8:00 AM - 6:00 PM')}
- Saturday-Sunday: Closed

The gym is closed on university holidays. Check recreation.ku.edu for current hours as they vary by semester."""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "hours"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 4. Chalk Rock climbing wall document
    chalk_rock = facilities.get("chalk_rock", {})
    if chalk_rock:
        hours_cr = chalk_rock.get("hours_spring_2025", {})
        
        doc = f"""The Chalk Rock - KU Climbing Wall

Location: Bottom floor of Ambler SRFC, Room #1
Height: {chalk_rock.get('height', '42 feet')}

Features:
- Main Wall: 6 belay lines, dynamic ropes, Gri-Gri belay devices
- Bouldering: 12-foot bouldering wall wrapping around two walls

Spring 2025 Hours:
- Sunday: {hours_cr.get('sunday', '5:00 PM - 8:00 PM')}
- Tuesday: {hours_cr.get('tuesday', '5:00 PM - 10:00 PM')}
- Wednesday: {hours_cr.get('wednesday', '5:00 PM - 10:00 PM')}
- Thursday: {hours_cr.get('thursday', '5:00 PM - 10:00 PM')}

Cost: FREE for students and faculty with valid KU ID
Requirements: Sign waiver, present valid KU ID

Belay Certification:
- Classes offered Sunday and Wednesday at 5pm
- Can register ahead or sign up upon arrival
- Short class duration

The climbing wall is free for KU students. Belay classes teach you how to safely belay other climbers."""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "facility", "name": "Chalk Rock"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 5. Outdoor facilities document
    outdoor = facilities.get("outdoor_facilities", {})
    if outdoor:
        shenk = outdoor.get("shenk_sports_complex", {})
        central = outdoor.get("central_field", {})
        
        doc = f"""KU Outdoor Recreation Facilities

Shenk Sports Complex:
- Location: 23rd and Iowa, Lawrence
- Named after: Henry Shenk (Chair of Health, Physical Education and Recreation 1941-1972)
- Features: 4 grass fields, soccer goals, restrooms, parking
- Usage: Intramural Sports, Sport Clubs, open recreation, reservations
- Contact: Kirsten King at kirstenking@ku.edu

Central Field:
- Location: Central District, between Downs Hall and Stouffer Place Apartments
- Surface: Outdoor artificial turf field
- Features: Movable soccer/lacrosse goals, electronic scoreboard, lighting (5am-11:30pm)
- Parking: Any Yellow Lot

Tennis and Sand Volleyball Courts:
- Location: Behind Ambler SRFC
- Features: Tennis courts, sand volleyball courts, outdoor basketball courts

All outdoor facilities are for KU students and employees. No vehicles or bikes on fields. No alcohol or glass containers."""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "outdoor_facilities"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 6. KU Fit group fitness document
    programs = recreation_data.get("programs", {})
    ku_fit = programs.get("ku_fit_group_exercise", {})
    if ku_fit:
        passes = ku_fit.get("passes", {})
        
        doc = f"""KU Fit Group Fitness Classes

{ku_fit.get('description', '')}

Classes Offered (approximately 35 per semester):
- Cardio: Cycling, HIIT, Dance, Cardio Mix
- Strength: Bodypump, Sculpt and Tone, Full Body Strength
- Mind/Body: Yoga, Pilates, Barre, Shapes
- Dance: Zumba, Latin Dance

Pass Options:
- Full Semester: ${passes.get('full_semester', {}).get('cost', '50')} (unlimited classes)
- Half Semester: ${passes.get('half_semester', {}).get('cost', '25')}
- Summer: ${passes.get('summer', {}).get('cost', '25')}
- Single Class: ${passes.get('one_class', {}).get('cost', '3')}

FREE classes during first week of semester and finals week!

Purchase at: ASRFC Admin Office (Room 103), Welcome Center, or recstore.ku.edu
No advance registration required - just show up!
Equipment: Yoga mats, blocks, and straps provided free for yoga classes"""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "program", "name": "KU Fit"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 7. Personal training document
    pt = programs.get("personal_training", {})
    if pt:
        packages = pt.get("packages", {})
        individual = packages.get("individual", {})
        
        doc = f"""KU Recreation Personal Training

{pt.get('description', '55-minute one-on-one training sessions with certified trainers')}

Services:
- Set short and long term fitness goals
- Design personalized exercise programs
- Provide motivation and accountability
- Teach proper form and technique
- Answer questions about exercise and nutrition

Individual Training Packages:
- Fit4U Assessment: ${individual.get('fit4u_assessment', {}).get('cost', '15')} (body composition + consultation)
- Starter Package: ${individual.get('starter', {}).get('cost', '30')} (assessment + 1 session)
- 5 Sessions: ${individual.get('5_sessions', {}).get('cost', '85')} (${individual.get('5_sessions', {}).get('per_session', '17')}/session)
- 10 Sessions: ${individual.get('10_sessions', {}).get('cost', '165')} (${individual.get('10_sessions', {}).get('per_session', '16.50')}/session)

Duo Training (with a friend): Starting at $12.50/person per session
Group Training (3-6 people): Starting at $7/person per session

Eligibility: Students, Faculty, Staff, Alumni with ASRFC membership
Contact: ptsrfc@ku.edu

Sign up at Admin Office (Room 103) or online. Trainer will contact you within 5 business days."""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "program", "name": "Personal Training"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 8. Intramural sports document
    im = programs.get("intramural_sports", {})
    if im:
        doc = f"""KU Intramural Sports

{im.get('description', 'Competitive sports opportunities in a safe, friendly environment')}

How to Join:
1. Purchase Intramural Sports Pass: ${im.get('pass', {}).get('cost', '15')}/semester
2. Register on IMLeagues.com or IMLeagues app
3. Bring valid KU ID to check in at games

Eligibility:
- Currently enrolled Lawrence or Edwards campus students
- Currently employed faculty and staff (must have Recreation Services membership)

Typical Sports Offered:
Flag football, Basketball, Volleyball, Soccer, Softball, Dodgeball, Tennis, Badminton, Table tennis, Racquetball

Key Policies:
- Maximum 15 players per team roster
- Can add players via IMLeagues during season
- $11.05 forfeit fine if minimum players not present
- Two forfeits = team dropped from event

Website: recreation.ku.edu/intramural-sports
Registration: IMLeagues.com"""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "program", "name": "Intramural Sports"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 9. Outdoor Pursuits equipment rental document
    odp = programs.get("outdoor_pursuits", {})
    if odp:
        rental = odp.get("equipment_rental", {})
        rates = rental.get("rates", {})
        
        doc = f"""Outdoor Pursuits Equipment Rental

Location: Room #1, bottom floor of ASRFC
Phone: {rental.get('contact', {}).get('phone', '785-864-1843')}
Email: {rental.get('contact', {}).get('email', 'outdoorpursuits@ku.edu')}

Reservations: Up to 14 days in advance; payment due at pickup
Winter Closure: November 11 - March 1 (rentals by appointment only)

Rental Rates:
Tents: 2-person $3/day, 4-person $4.50/day, 5-6 person $6/day
Watercraft: Canoe $9/day, Paddleboard $7/day, Kayak $9/day
Sleep Gear: Sleeping bag $2/day, Sleeping pad $1/day
Other: Backpack $3/day, Hammock $2/day, Cooler $3/day, Climbing shoes $5/day

Water Accessories: Life jacket $1/day, Dry bags $0.75-$5/day

Fees:
- Late fee: $5 per day per item
- Cleaning fee: $20 if not returned clean with all parts

Policies: Tents must be set up at pickup and return to verify condition."""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "program", "name": "Equipment Rental"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 10. Sport Clubs overview document
    sport_clubs = programs.get("sport_clubs", {})
    if sport_clubs:
        doc = f"""KU Sport Clubs

{sport_clubs.get('description', 'Student-led organizations for competitive and recreational sports')}

Structure: Student-organized, student-managed, student-operated with KU Recreation Services support

30+ Active Sport Clubs Including:
- Combat/Martial Arts: Fencing, Jiu Jitsu, Tae Kwon Do
- Team Sports: Baseball, Ice Hockey, Men's/Women's Lacrosse, Men's/Women's Rugby, Men's/Women's Soccer, Softball, Men's/Women's Volleyball
- Individual Sports: Badminton, Disc Golf, Golf, Running, Sailing, Swimming, Table Tennis, Tennis, Water Skiing
- Outdoor/Adventure: Rock Climbing, Crew (Rowing)
- Unique Sports: Quadball (formerly Quidditch), Roundnet (Spikeball), Dance Alliance, Ultimate Frisbee (Men's Horror-Zontals, Women's Betty)

How to Join:
1. Find clubs at recreation.ku.edu/current-sport-clubs
2. Contact club directly or attend practices/tryouts
3. Pay dues (minimum $5, varies by club: $20-$50 for non-traveling, several hundred for traveling clubs)

Tryouts: Typically held at beginning of each semester for competitive clubs
Registration: DoSportsEasyKU and Rock Chalk Central"""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "program", "name": "Sport Clubs Overview"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 11. Individual sport club documents
    clubs = sport_clubs.get("current_clubs", [])
    for club in clubs:
        doc = f"""Sport Club: {club.get('name', '')}

{club.get('description', '')}"""
        
        if club.get('practice'):
            doc += f"\nPractice: {club.get('practice')}"
        if club.get('games'):
            doc += f"\nGames: {club.get('games')}"
        
        doc += "\n\nTo join, contact the club directly or visit recreation.ku.edu/current-sport-clubs"
        
        documents.append(doc)
        metadatas.append({
            "source": "recreation",
            "type": "sport_club",
            "name": club.get("name", "")
        })
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 12. Memberships document
    memberships = recreation_data.get("memberships", {})
    if memberships:
        students = memberships.get("students", {})
        faculty = memberships.get("faculty_staff", {})
        alumni = memberships.get("alumni", {})
        guests = memberships.get("guests", {})
        
        doc = f"""KU Recreation Membership Information

STUDENTS:
- Currently enrolled in 3+ credit hours: INCLUDED in Wellness Student Fee (automatic!)
- Off-term students: ${memberships.get('students', {}).get('off_term_students', {}).get('cost', {}).get('monthly', '25')}/month (1 term max)
- Summer (not enrolled): ${memberships.get('students', {}).get('summer_memberships', {}).get('cost', {}).get('monthly', '25')}/month

FACULTY & STAFF:
- Weekly: ${faculty.get('cost', {}).get('weekly', '6.25')}
- Monthly: ${faculty.get('cost', {}).get('monthly', '25')}
- Annual: ${faculty.get('cost', {}).get('annual', '300')}
- FREE one-week trial available!

SPOUSE/DOMESTIC PARTNER:
- Same rates as faculty/staff: $6.25/week, $25/month, $300/year

ALUMNI (KU Alumni Association members):
- Monthly: ${alumni.get('cost', {}).get('monthly', '29.17')}
- Annual: ${alumni.get('cost', {}).get('annual', '350')}

GUESTS:
- Cost: ${guests.get('cost', '$7.02/day')}
- Must be sponsored by current ASRFC member
- Sponsor must be present with guest
- Guest must be 18+ with photo ID

Purchase Locations:
- Admin Office (Room 103) Mon-Fri 3-6pm
- Welcome Center
- Online: recstore.ku.edu

Payment: Credit card or check only (NO CASH)"""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "memberships"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 13. Aquatics document
    aquatics = recreation_data.get("aquatics", {})
    if aquatics:
        alternatives = aquatics.get("alternatives", {})
        indoor = alternatives.get("indoor", {})
        
        doc = f"""Swimming and Pool Access at KU

IMPORTANT: Robinson Center pools are CLOSED to students, faculty, and staff for open swim as of June 2022.

The closure was due to the Department of Health, Sport & Exercise Sciences no longer providing swim programming and difficulty hiring lifeguards.

KU Swimming & Diving varsity teams continue to use Robinson Natatorium for training.

ALTERNATIVES FOR SWIMMING:

Lawrence Indoor Aquatic Center:
- Address: {indoor.get('address', '4706 Overland Drive, Lawrence')}
- Operator: Lawrence Parks and Recreation
- Cost: {indoor.get('cost', '$6 adults (18-59), $5 youth/seniors')}

Lawrence Outdoor Aquatic Center:
- Location: Off Kentucky Street
- Operator: Lawrence Parks and Recreation
- Seasonal operation

For swimming needs, use Lawrence Parks and Recreation facilities."""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "aquatics"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 14. FAQs
    faqs = recreation_data.get("faqs", [])
    for faq in faqs:
        doc = f"""Recreation FAQ: {faq.get('question', '')}

{faq.get('answer', '')}"""
        
        documents.append(doc)
        metadatas.append({
            "source": "recreation",
            "type": "faq",
            "question": faq.get("question", "")
        })
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    # 15. Contact information document
    contact = recreation_data.get("contact", {})
    if contact:
        main = contact.get("main", {})
        
        doc = f"""KU Recreation Contact Information

Main Office:
Phone: {main.get('phone', '785-864-3546')}
Email: {main.get('email', 'recreation@ku.edu')}
Address: {main.get('address', '1740 Watkins Center Dr, Lawrence, KS 66045')}

Outdoor Pursuits:
Phone: {contact.get('outdoor_pursuits', {}).get('phone', '785-864-1843')}
Email: {contact.get('outdoor_pursuits', {}).get('email', 'outdoorpursuits@ku.edu')}

Personal Training:
Email: {contact.get('personal_training', {}).get('email', 'ptsrfc@ku.edu')}

KU Fit Group Fitness:
Email: {contact.get('ku_fit', {}).get('email', 'kufit@ku.edu')}

Memberships:
Phone: {contact.get('memberships', {}).get('phone', '785-864-1370')}

Facility Reservations:
Contact: {contact.get('facility_reservations', {}).get('contact', 'Kirsten King')}
Email: {contact.get('facility_reservations', {}).get('email', 'kirstenking@ku.edu')}

Website: recreation.ku.edu
Social Media: @kuamblerrec (Instagram, Twitter, YouTube)"""

        documents.append(doc)
        metadatas.append({"source": "recreation", "type": "contact"})
        ids.append(f"rec_{doc_counter}")
        doc_counter += 1
    
    return documents, metadatas, ids


def prepare_campus_safety_documents(safety_data: Dict[str, Any]) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert campus safety JSON data into documents for embedding.
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    # 1. Overview document
    overview = safety_data.get("overview", {})
    overview_text = f"""KU Campus Safety Overview
        {overview.get('description', '')}
        Mission: {overview.get('mission', '')}
        Philosophy: {overview.get('philosophy', '')}
        Campus Size: {overview.get('campus_size', '')}
        Website: {overview.get('website', '')}"""
    
    documents.append(overview_text)
    metadatas.append({"source": "campus_safety", "type": "overview"})
    ids.append("safety_overview")
    
    # 2. Emergency contacts document
    contacts = safety_data.get("emergency_contacts", {})
    contacts_text = "KU Emergency Contacts\n\n"
    for contact_name, contact_info in contacts.items():
        if isinstance(contact_info, dict):
            readable_name = contact_name.replace("_", " ").title()
            number = contact_info.get("number", contact_info.get("daytime", ""))
            desc = contact_info.get("description", "")
            contacts_text += f"{readable_name}: {number}"
            if desc:
                contacts_text += f" - {desc}"
            contacts_text += "\n"
            # Add additional numbers if present
            if "after_hours" in contact_info:
                contacts_text += f"  After Hours: {contact_info['after_hours']}\n"
            if "email" in contact_info:
                contacts_text += f"  Email: {contact_info['email']}\n"
    
    documents.append(contacts_text)
    metadatas.append({"source": "campus_safety", "type": "emergency_contacts"})
    ids.append("safety_emergency_contacts")
    
    # 3. KU Police Department document
    kupd = safety_data.get("ku_police_department", {})
    kupd_location = kupd.get("location", {})
    kupd_contact = kupd.get("contact", {})
    
    kupd_text = f"""KU Police Department (KUPD)
        Location: {kupd_location.get('building', '')}, {kupd_location.get('address', '')}, {kupd_location.get('city', '')}
        Bus Routes: {', '.join(kupd_location.get('bus_routes', []))}

        Contact Information:
        Main Number: {kupd_contact.get('main_number', '')}
        Fax: {kupd_contact.get('fax', '')}
        Email: {kupd_contact.get('email', '')}

        Department Units:
        """
    for unit in kupd.get("units", []):
        kupd_text += f"- {unit.get('name', '')}: {unit.get('description', '')}\n"
    
    # Add crime statistics
    stats = kupd.get("statistics", {})
    if stats:
        kupd_text += f"""
        Crime Statistics:
        - 2024 Crimes Reported: {stats.get('2024_crimes_reported', '')}
        - Change from 2023: {stats.get('change_from_2023', '')}
        - 10-Year Average: {stats.get('ten_year_average', '')}
        - Most Common: {', '.join(stats.get('most_common_crimes', []))}
        - Daily Crime Log: {stats.get('daily_crime_log', '')}"""
    
    documents.append(kupd_text)
    metadatas.append({"source": "campus_safety", "type": "police_department"})
    ids.append("safety_kupd")
    
    # 4. Safety Services documents
    services = safety_data.get("safety_services", {})
    
    # Security Escorts
    escorts = services.get("security_escorts", {})
    if escorts:
        escort_text = f"""KU Security Escorts
        {escorts.get('description', '')}
        Phone: {escorts.get('phone', '')}
        Availability: {escorts.get('availability', '')}
        Request a safety escort from campus facilities to parking lots or on-campus living facilities."""
        documents.append(escort_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "security_escorts"})
        ids.append("safety_service_escorts")
    
    # SafeBus
    safebus = services.get("safebus", {})
    if safebus:
        safebus_text = f"""KU SafeBus Late Night Transportation
        {safebus.get('description', '')}
        Hours: Until {safebus.get('hours', '')}
        Days: {safebus.get('days', '')}
        Route Number: {safebus.get('route_number', '')}
        Tracking: Use {safebus.get('tracking', '')} app
        Note: {safebus.get('note', '')}"""
        documents.append(safebus_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "safebus"})
        ids.append("safety_service_safebus")
    
    # SafeRide (discontinued)
    saferide = services.get("saferide", {})
    if saferide:
        saferide_text = f"""KU SafeRide Service
        Status: {saferide.get('status', '')}
        SafeRide was a free transportation service for KU students providing safe rides home.
        Previous service details:
        - Hours: {saferide.get('previous_service', {}).get('hours', '')}
        - Days: {saferide.get('previous_service', {}).get('days', '')}
        - Phone: {saferide.get('previous_service', {}).get('phone', '')}
        - Service Area: {saferide.get('previous_service', {}).get('service_area', '')}
        - Started: {saferide.get('previous_service', {}).get('started', '')}
        Use SafeBus for late-night transportation instead."""
        documents.append(saferide_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "saferide"})
        ids.append("safety_service_saferide")
    
    # Lost and Found
    lost_found = services.get("lost_and_found", {})
    if lost_found:
        lost_text = f"""KU Lost and Found
        {lost_found.get('description', '')}
        Email: {lost_found.get('email', '')}
        Retention Period: {lost_found.get('retention_period', '')}
        Pickup Hours: {lost_found.get('pickup_hours', '')}
        Location: {lost_found.get('location', '')}"""
        documents.append(lost_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "lost_and_found"})
        ids.append("safety_service_lost_found")
    
    # Fingerprinting
    fingerprint = services.get("fingerprinting", {})
    if fingerprint:
        fp_text = f"""KU Fingerprinting Service
        Availability: {fingerprint.get('availability', '')}
        Phone: {fingerprint.get('phone', '')}
        Cost: {fingerprint.get('cost', '')}
        Payment: {fingerprint.get('payment', '')}
        Types: {', '.join(fingerprint.get('types', []))}
        Eligibility: {fingerprint.get('eligibility', '')}"""
        documents.append(fp_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "fingerprinting"})
        ids.append("safety_service_fingerprinting")
    
    # Weapons Storage
    weapons = services.get("weapons_storage", {})
    if weapons:
        weapons_text = f"""KU Weapons Storage Service
        {weapons.get('description', '')}
        Hours: {weapons.get('hours', '')}
        Requirements:
        {chr(10).join('- ' + r for r in weapons.get('requirements', []))}
        Policy:
        {chr(10).join('- ' + p for p in weapons.get('policy', []))}"""
        documents.append(weapons_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "weapons_storage"})
        ids.append("safety_service_weapons")
    
    # Bicycle Registration
    bike = services.get("bicycle_registration", {})
    if bike:
        bike_text = f"""KU Bicycle Registration
        {bike.get('description', '')}
        Phone: {bike.get('phone', '')}
        Contact: {bike.get('contact', '')}
        Serial Number Required: {'Yes' if bike.get('serial_number_required') else 'No'}
        Marking Service: {bike.get('marking_service', '')}"""
        documents.append(bike_text)
        metadatas.append({"source": "campus_safety", "type": "service", "name": "bicycle_registration"})
        ids.append("safety_service_bicycle")
    
    # 5. Blue Light Phones
    blue_light = safety_data.get("blue_light_phones", {})
    if blue_light:
        blue_text = f"""KU Blue Light Emergency Phones
        Status: {blue_light.get('status', '')}
        {blue_light.get('description', '')}
        History: {blue_light.get('history', '')}
        Count: {blue_light.get('count', '')}
        Function: {blue_light.get('function', '')}
        Phase Out Reason: {blue_light.get('phase_out_reason', '')}
        Alternatives Being Considered: {', '.join(blue_light.get('alternatives_being_considered', []))}
        Note: {blue_light.get('note', '')}"""
        documents.append(blue_text)
        metadatas.append({"source": "campus_safety", "type": "blue_light_phones"})
        ids.append("safety_blue_light")
    
    # 6. AED Program
    aed = safety_data.get("aed_program", {})
    if aed:
        aed_text = f"""KU AED (Automated External Defibrillator) Program
        {aed.get('description', '')}
        Count: {aed.get('count', '')}
        Locations: {aed.get('locations', '')}
        Most Common Models: {', '.join(aed.get('most_common_models', []))}
        Inspection: {aed.get('inspection', '')}
        Police AEDs: {aed.get('police_aeds', '')}
        Training: {aed.get('training', '')}
        PulsePoint App: {aed.get('pulsepoint_app', '')}
        AED Map: {aed.get('aed_map', '')}

        Status Indicator:
        - Green Flash: {aed.get('status_indicator', {}).get('green_flash', '')}
        - Red/Orange Flash or Beeping: {aed.get('status_indicator', {}).get('red_or_orange_flash_or_beeping', '')}"""
        documents.append(aed_text)
        metadatas.append({"source": "campus_safety", "type": "aed_program"})
        ids.append("safety_aed")
    
    # 7. Emergency Notification Systems
    notifications = safety_data.get("emergency_notification_systems", {})
    if notifications:
        notif_text = "KU Emergency Notification Systems\n\n"
        for system_name, system_info in notifications.items():
            if isinstance(system_info, dict):
                readable_name = system_name.replace("_", " ").title()
                notif_text += f"{readable_name}:\n"
                notif_text += f"  {system_info.get('description', '')}\n"
                if 'website' in system_info:
                    notif_text += f"  Website: {system_info['website']}\n"
                if 'number' in system_info:
                    notif_text += f"  Number: {system_info['number']}\n"
                notif_text += "\n"
        documents.append(notif_text)
        metadatas.append({"source": "campus_safety", "type": "emergency_notifications"})
        ids.append("safety_notifications")
    
    # 8. Safety Tips
    tips = safety_data.get("safety_tips", {})
    if tips:
        tips_text = "KU Campus Safety Tips\n\n"
        for category, tip_list in tips.items():
            readable_cat = category.replace("_", " ").title()
            tips_text += f"{readable_cat}:\n"
            for tip in tip_list:
                tips_text += f"- {tip}\n"
            tips_text += "\n"
        documents.append(tips_text)
        metadatas.append({"source": "campus_safety", "type": "safety_tips"})
        ids.append("safety_tips")
    
    # 9. Concealed Carry
    cc = safety_data.get("concealed_carry", {})
    if cc:
        cc_text = f"""KU Concealed Carry Policy
        Effective Date: {cc.get('effective_date', '')}
        Law: {cc.get('law', '')}
        Website: {cc.get('website', '')}

        Age Requirements:
        - 21 and older: {cc.get('age_requirements', {}).get('21_and_older', '')}
        - 18 to 20: {cc.get('age_requirements', {}).get('18_to_20', '')}

        Where Allowed: {cc.get('where_allowed', '')}

        Where Prohibited:
        {chr(10).join('- ' + p for p in cc.get('where_prohibited', []))}

        Requirements:
        {chr(10).join('- ' + r for r in cc.get('requirements', []))}

        Storage Options:
        {chr(10).join('- ' + s for s in cc.get('storage_options', []))}

        Prohibited Actions:
        {chr(10).join('- ' + p for p in cc.get('prohibited_actions', []))}

        Violations: {cc.get('violations', '')}

        Important Notes:
        {chr(10).join('- ' + n for n in cc.get('important_notes', []))}"""
        documents.append(cc_text)
        metadatas.append({"source": "campus_safety", "type": "concealed_carry"})
        ids.append("safety_concealed_carry")
    
    # 10. CCTV System
    cctv = safety_data.get("cctv_system", {})
    if cctv:
        cctv_text = f"""KU CCTV Camera System
        {cctv.get('description', '')}
        Capabilities:
        {chr(10).join('- ' + c for c in cctv.get('capabilities', []))}
        Monitoring: {cctv.get('monitoring', '')}
        Locations: {cctv.get('locations', '')}
        Policy: {cctv.get('policy', '')}"""
        documents.append(cctv_text)
        metadatas.append({"source": "campus_safety", "type": "cctv"})
        ids.append("safety_cctv")
    
    # 11. Clery Act
    clery = safety_data.get("clery_act", {})
    if clery:
        clery_text = f"""Clery Act and Campus Crime Statistics
        {clery.get('description', '')}
        Report Name: {clery.get('report_name', '')}
        Report Availability: {clery.get('report_availability', '')}
        Statistics Location: {clery.get('statistics_location', '')}
        Daily Crime Log: {clery.get('daily_crime_log', '')}
        Ten Year Statistics: {clery.get('ten_year_statistics', '')}"""
        documents.append(clery_text)
        metadatas.append({"source": "campus_safety", "type": "clery_act"})
        ids.append("safety_clery")
    
    # 12. Reporting Information
    reporting = safety_data.get("reporting", {})
    if reporting:
        report_text = "KU Safety Reporting Information\n\n"
        
        crime_report = reporting.get("crime_reporting", {})
        if crime_report:
            report_text += f"Crime Reporting:\n"
            report_text += f"Policy: {crime_report.get('policy', '')}\n"
            report_text += "How to Report:\n"
            for method in crime_report.get("how_to_report", []):
                report_text += f"- {method}\n"
            report_text += "\n"
        
        sa_report = reporting.get("sexual_assault_harassment", {})
        if sa_report:
            report_text += f"Sexual Assault/Harassment:\n"
            report_text += f"Report to: {sa_report.get('report_to', '')}\n"
            report_text += f"Website: {sa_report.get('website', '')}\n\n"
        
        safety_concerns = reporting.get("safety_concerns", {})
        if safety_concerns:
            report_text += f"Safety Concerns:\n"
            report_text += f"Contact: {safety_concerns.get('contact', '')}\n"
            report_text += f"Email: {safety_concerns.get('email', '')}\n"
            report_text += f"Phone: {safety_concerns.get('phone', '')}\n"
            report_text += f"Anonymous Reporting: {safety_concerns.get('anonymous_reporting', '')}\n"
        
        documents.append(report_text)
        metadatas.append({"source": "campus_safety", "type": "reporting"})
        ids.append("safety_reporting")
    
    # 13. FAQs
    faqs = safety_data.get("faqs", [])
    for i, faq in enumerate(faqs):
        faq_text = f"""Campus Safety FAQ: {faq.get('question', '')}
            {faq.get('answer', '')}"""
        documents.append(faq_text)
        metadatas.append({
            "source": "campus_safety",
            "type": "faq",
            "question": faq.get("question", "")
        })
        ids.append(f"safety_faq_{i+1}")
    
    return documents, metadatas, ids


def prepare_student_orgs_documents(orgs_data: Dict[str, Any]) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Convert student organizations JSON data into documents for embedding.
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    # 1. Overview document
    overview = orgs_data.get("overview", {})
    sec = overview.get("student_engagement_center", {})
    overview_text = f"""KU Student Organizations Overview
        {overview.get('description', '')}

        Registration Platform: {overview.get('registration_platform', '')}
        Governing Office: {overview.get('governing_office', '')}
        Location: {overview.get('location', '')}
        Phone: {overview.get('phone', '')}
        Email: {overview.get('email', '')}

        Student Engagement Center:
        {sec.get('description', '')}
        Location: {sec.get('location', '')}
        Phone: {sec.get('phone', '')}
        Email: {sec.get('email', '')}"""
    
    documents.append(overview_text)
    metadatas.append({"source": "student_organizations", "type": "overview"})
    ids.append("orgs_overview")
    
    # 2. Rock Chalk Central document
    rcc = orgs_data.get("rock_chalk_central", {})
    if rcc:
        rcc_text = f"""Rock Chalk Central - KU Student Organization Database
        {rcc.get('description', '')}
        URL: {rcc.get('url', '')}
        Mobile App: {rcc.get('mobile_app', '')}
        Login: {rcc.get('login', '')}

        Features:
        {chr(10).join('- ' + f for f in rcc.get('features', []))}"""
        documents.append(rcc_text)
        metadatas.append({"source": "student_organizations", "type": "platform"})
        ids.append("orgs_rock_chalk_central")
    
    # 3. Organization Categories
    categories = orgs_data.get("organization_categories", [])
    if categories:
        cat_text = f"""KU Student Organization Categories
        KU has over 600 registered student organizations in the following categories:

        {chr(10).join('- ' + c for c in categories)}

        Browse organizations by category on Rock Chalk Central (rockchalkcentral.ku.edu)."""
        documents.append(cat_text)
        metadatas.append({"source": "student_organizations", "type": "categories"})
        ids.append("orgs_categories")
    
    # 4. Getting Involved
    involved = orgs_data.get("getting_involved", {})
    if involved:
        fairs = involved.get("involvement_fairs", {})
        involved_text = f"""Getting Involved at KU

        Involvement Fairs:

        UnionFest (Fall):
        - Timing: {fairs.get('unionfest', {}).get('timing', '')}
        - {fairs.get('unionfest', {}).get('description', '')}
        - Participants: {fairs.get('unionfest', {}).get('participants', '')}

        Spring Involvement Fair:
        - Timing: {fairs.get('spring_fair', {}).get('timing', '')}
        - {fairs.get('spring_fair', {}).get('description', '')}

        Recommendation: {involved.get('recommendation', '')}

        Exploration Areas:
        {chr(10).join('- ' + a for a in involved.get('exploration_areas', []))}"""
        documents.append(involved_text)
        metadatas.append({"source": "student_organizations", "type": "getting_involved"})
        ids.append("orgs_getting_involved")
    
    # 5. Starting an Organization
    starting = orgs_data.get("starting_organization", {})
    if starting:
        start_text = f"""Starting a New Student Organization at KU

        Requirements:
        {chr(10).join('- ' + r for r in starting.get('requirements', []))}

        Registration Period: {starting.get('registration_period', '')}
        Appeal Process: {starting.get('appeal_process', '')}

        Register through Rock Chalk Central (rockchalkcentral.ku.edu)."""
        documents.append(start_text)
        metadatas.append({"source": "student_organizations", "type": "starting_organization"})
        ids.append("orgs_starting")
    
    # 6. Student Senate
    senate = orgs_data.get("student_senate", {})
    if senate:
        comp = senate.get("composition", {})
        senate_text = f"""KU Student Senate
        {senate.get('description', '')}

        Composition:
        - Senators: {comp.get('senators', '')}
        - Legislative Officers: {comp.get('legislative_officers', '')}
        - Executive Staff: {comp.get('executive_staff', '')}
        - Election Commission: {comp.get('election_commission', '')}

        Budget: {senate.get('budget', '')}
        Student Fee: {senate.get('student_fee', '')}
        Website: {senate.get('website', '')}

        Functions:
        {chr(10).join('- ' + f for f in senate.get('functions', []))}

Get Involved: {senate.get('involvement', '')}"""
        documents.append(senate_text)
        metadatas.append({"source": "student_organizations", "type": "student_senate"})
        ids.append("orgs_student_senate")
    
    # 7. Student Union Activities (SUA)
    sua = orgs_data.get("student_union_activities", {})
    if sua:
        sua_text = f"""Student Union Activities (SUA)
        {sua.get('description', '')}

        History: {sua.get('history', '')}
        Events Per Year: {sua.get('events_per_year', '')}
        Attendance Example: {sua.get('attendance_example', '')}
        Website: {sua.get('website', '')}
        Committees: {sua.get('committees', '')}

        Event Types:
        {chr(10).join('- ' + e for e in sua.get('event_types', []))}

        Notable Events:
        {chr(10).join('- ' + e for e in sua.get('notable_events', []))}

        How to Join: {sua.get('how_to_join', '')}"""
        documents.append(sua_text)
        metadatas.append({"source": "student_organizations", "type": "sua"})
        ids.append("orgs_sua")
    
    # 8. Greek Life Overview
    greek = orgs_data.get("sorority_and_fraternity_life", {})
    if greek:
        greek_overview = greek.get("overview", {})
        greek_text = f"""KU Sorority & Fraternity Life Overview
        History: {greek_overview.get('history', '')}
        Total Members: {greek_overview.get('total_members', '')}
        Total Organizations: {greek_overview.get('total_organizations', '')}
        Governing Councils: {greek_overview.get('governing_councils', '')}

        Office Location: {greek_overview.get('office_location', '')}
        Phone: {greek_overview.get('phone', '')}
        Email: {greek_overview.get('email', '')}
        Website: {greek_overview.get('website', '')}

        Core Values:
        {chr(10).join('- ' + v for v in greek.get('core_values', []))}

        Mission: {greek.get('mission', '')}"""
        documents.append(greek_text)
        metadatas.append({"source": "student_organizations", "type": "greek_overview"})
        ids.append("orgs_greek_overview")
    
    # 9. IFC (Interfraternity Council)
    councils = greek.get("governing_councils", {})
    ifc = councils.get("ifc", {})
    if ifc:
        ifc_text = f"""KU Interfraternity Council (IFC)
        {ifc.get('description', '')}

        Chapters: {ifc.get('chapters', '')}
        Housed Chapters: {ifc.get('housed_chapters', '')}
        Website: {ifc.get('website', '')}
        Email: {ifc.get('email', '')}

        IFC Fraternities:
        {chr(10).join('- ' + c for c in ifc.get('chapters_list', []))}

        Recruitment:
        Structured Recruitment: {ifc.get('recruitment', {}).get('structured', {}).get('description', '')}
        - Timing: {ifc.get('recruitment', {}).get('structured', {}).get('timing', '')}

        Unstructured Recruitment: {ifc.get('recruitment', {}).get('unstructured', {}).get('description', '')}
        - Timing: {ifc.get('recruitment', {}).get('unstructured', {}).get('timing', '')}

        Housing:
        {ifc.get('housing', {}).get('description', '')}
        Amenities: {', '.join(ifc.get('housing', {}).get('amenities', [])[:5])}..."""
        documents.append(ifc_text)
        metadatas.append({"source": "student_organizations", "type": "greek_ifc"})
        ids.append("orgs_greek_ifc")
    
    # 10. Individual IFC chapters
    for chapter in ifc.get("chapters_list", []):
        chapter_text = f"""{chapter} - KU IFC Fraternity
        {chapter} is a member of the KU Interfraternity Council (IFC).

        For more information, visit kuifc.org or contact chapters directly.
        Recruitment: Year-round through structured and unstructured processes.
        Contact IFC at kuifc@ku.edu for recruitment information."""
        documents.append(chapter_text)
        metadatas.append({
            "source": "student_organizations",
            "type": "greek_chapter",
            "council": "IFC",
            "name": chapter
        })
        ids.append(f"orgs_ifc_{chapter.lower().replace(' ', '_')}")
    
    # 11. PHA (Panhellenic Association)
    pha = councils.get("pha", {})
    if pha:
        ffr = pha.get("recruitment", {}).get("fall_formal_recruitment", {})
        ffr_dates = ffr.get("2025_dates", {})
        pha_text = f"""KU Panhellenic Association (PHA)
        Founded: {pha.get('founded', '')}
        Founding Purpose: {pha.get('founding_purpose', '')}
        Chapters: {pha.get('chapters', '')}
        Website: {pha.get('website', '')}

        Pillars: {', '.join(pha.get('pillars', []))}
        Governed By: {pha.get('governed_by', '')}

        PHA Sororities:
        {chr(10).join('- ' + c for c in pha.get('chapters_list', []))}

        Fall Formal Recruitment 2025:
        - Registration: {ffr_dates.get('registration', '')}
        - Move-in/Orientation: {ffr_dates.get('move_in_orientation', '')}
        - Open House: {ffr_dates.get('open_house', '')}
        - Philanthropy: {ffr_dates.get('philanthropy', '')}
        - Sisterhood: {ffr_dates.get('sisterhood', '')}
        - Preference: {ffr_dates.get('preference', '')}
        - Bid Day: {ffr_dates.get('bid_day', '')}

        Registration Fee: {ffr.get('registration_fee', '')}
        Typical Participants: {ffr.get('participants', '')}

        Continuous Open Recruitment:
        {pha.get('recruitment', {}).get('continuous_open_recruitment', {}).get('timing', '')}"""
        documents.append(pha_text)
        metadatas.append({"source": "student_organizations", "type": "greek_pha"})
        ids.append("orgs_greek_pha")
    
    # 12. Individual PHA chapters
    for chapter in pha.get("chapters_list", []):
        chapter_text = f"""{chapter} - KU Panhellenic Sorority
        {chapter} is a member of the KU Panhellenic Association (PHA).

        Join through Fall Formal Recruitment (August) or Continuous Open Recruitment.
        Website: kupanhellenic.com
        Registration Fee: $240 for formal recruitment."""
        documents.append(chapter_text)
        metadatas.append({
            "source": "student_organizations",
            "type": "greek_chapter",
            "council": "PHA",
            "name": chapter
        })
        ids.append(f"orgs_pha_{chapter.lower().replace(' ', '_')}")
    
    # 13. NPHC (National Pan-Hellenic Council)
    nphc = councils.get("nphc", {})
    if nphc:
        nphc_text = f"""KU National Pan-Hellenic Council (NPHC)
        {nphc.get('description', '')}

        Chapters: {nphc.get('chapters', '')}
        Website: {nphc.get('website', '')}

        NPHC Organizations:
        {chr(10).join('- ' + c for c in nphc.get('chapters_list', []))}

        Joining Process: {nphc.get('joining', {}).get('process', '')}
        How to Start: {nphc.get('joining', {}).get('how_to_start', '')}
        Events: {', '.join(nphc.get('joining', {}).get('events', []))}
        Note: {nphc.get('joining', {}).get('note', '')}"""
        documents.append(nphc_text)
        metadatas.append({"source": "student_organizations", "type": "greek_nphc"})
        ids.append("orgs_greek_nphc")
    
    # 14. Individual NPHC chapters
    for chapter in nphc.get("chapters_list", []):
        chapter_text = f"""{chapter} - KU NPHC Organization
        {chapter} is a historically Black Greek letter organization and member of the KU National Pan-Hellenic Council (NPHC).

        Joining: Through Membership Intake process
        How to Start: Attend NPHC Week events including Meet The Greeks
        Website: kunphc.com"""
        documents.append(chapter_text)
        metadatas.append({
            "source": "student_organizations",
            "type": "greek_chapter",
            "council": "NPHC",
            "name": chapter
        })
        ids.append(f"orgs_nphc_{chapter.lower().replace(' ', '_').replace('.', '').replace(',', '')[:30]}")
    
    # 15. MGC (Multicultural Greek Council)
    mgc = councils.get("mgc", {})
    if mgc:
        mgc_text = f"""KU Multicultural Greek Council (MGC)
        {mgc.get('description', '')}

        Chapters: {mgc.get('chapters', '')}
        Website: {mgc.get('website', '')}

        Purpose:
        {chr(10).join('- ' + p for p in mgc.get('purpose', []))}

        MGC Organizations:
        {chr(10).join('- ' + c for c in mgc.get('chapters_list', []))}

        Joining Process: {mgc.get('joining', {}).get('process', '')}
        How to Start: {mgc.get('joining', {}).get('how_to_start', '')}"""
        documents.append(mgc_text)
        metadatas.append({"source": "student_organizations", "type": "greek_mgc"})
        ids.append("orgs_greek_mgc")
    
    # 16. Individual MGC chapters
    for chapter in mgc.get("chapters_list", []):
        chapter_text = f"""{chapter} - KU MGC Organization
        {chapter} is a culturally-based organization and member of the KU Multicultural Greek Council (MGC).

        Joining: Through Membership Intake process
        How to Start: Attend MGC Week events
        Website: kumgc.com"""
        documents.append(chapter_text)
        metadatas.append({
            "source": "student_organizations",
            "type": "greek_chapter",
            "council": "MGC",
            "name": chapter
        })
        ids.append(f"orgs_mgc_{chapter.lower().replace(' ', '_').replace('.', '').replace(',', '')[:30]}")
    
    # 17. Greek Programs
    programs = greek.get("programs", {})
    if programs:
        rcr = programs.get("rock_chalk_revue", {})
        sfl = programs.get("sfl_advance", {})
        prog_text = f"""KU Greek Life Programs

        Rock Chalk Revue:
        {rcr.get('description', '')}
        Format: {rcr.get('format', '')}
        Awards: {rcr.get('awards', '')}

        SFL Advance:
        {sfl.get('description', '')}
        Focus: {', '.join(sfl.get('focus', []))}"""
        documents.append(prog_text)
        metadatas.append({"source": "student_organizations", "type": "greek_programs"})
        ids.append("orgs_greek_programs")
    
    # 18. Greek Costs
    costs = greek.get("costs", {})
    if costs:
        cost_text = f"""Greek Life Costs at KU
        Dues Range: {costs.get('dues_range', '')}
        Typical Range: {costs.get('typical_range', '')}
        Transparency: {costs.get('transparency', '')}"""
        documents.append(cost_text)
        metadatas.append({"source": "student_organizations", "type": "greek_costs"})
        ids.append("orgs_greek_costs")
    
    # 19. Major Campus Organizations
    major = orgs_data.get("major_campus_organizations", {})
    if major:
        major_text = "Major Campus Organizations at KU\n\n"
        for org_name, org_info in major.items():
            if isinstance(org_info, dict):
                readable_name = org_info.get('name', org_name.replace('_', ' ').title())
                major_text += f"{readable_name}:\n"
                major_text += f"  {org_info.get('description', '')}\n"
                if 'broadcast' in org_info:
                    major_text += f"  Broadcast: {org_info['broadcast']}\n"
                if 'participants' in org_info:
                    major_text += f"  Participants: {org_info['participants']}\n"
                major_text += "\n"
        documents.append(major_text)
        metadatas.append({"source": "student_organizations", "type": "major_organizations"})
        ids.append("orgs_major")
    
    # 20. Cultural Organizations
    cultural = orgs_data.get("cultural_organizations", {})
    if cultural:
        cultural_text = f"""Cultural Organizations at KU
        Examples of cultural and identity organizations:
        {chr(10).join('- ' + o for o in cultural.get('examples', []))}

        Find more cultural organizations on Rock Chalk Central (rockchalkcentral.ku.edu)."""
        documents.append(cultural_text)
        metadatas.append({"source": "student_organizations", "type": "cultural_organizations"})
        ids.append("orgs_cultural")
    
    # 21. Academic/Professional Organizations
    academic = orgs_data.get("academic_professional_organizations", {})
    if academic:
        academic_text = "Academic and Professional Organizations at KU\n\n"
        for field, orgs in academic.items():
            if isinstance(orgs, list):
                readable_field = field.replace('_', ' ').title()
                academic_text += f"{readable_field}:\n"
                for org in orgs:
                    academic_text += f"- {org}\n"
                academic_text += "\n"
        academic_text += "Find more academic organizations on Rock Chalk Central."
        documents.append(academic_text)
        metadatas.append({"source": "student_organizations", "type": "academic_organizations"})
        ids.append("orgs_academic")
    
    # 22. Identity/Affinity Organizations
    identity = orgs_data.get("identity_affinity_organizations", {})
    if identity:
        identity_text = "Identity and Affinity Organizations at KU\n\n"
        for group, info in identity.items():
            if isinstance(info, dict):
                identity_text += f"{info.get('name', group)}:\n"
                if 'resource' in info:
                    identity_text += f"  Resource: {info['resource']}\n"
                if 'location' in info:
                    identity_text += f"  Location: {info['location']}\n"
                if 'focus' in info:
                    identity_text += f"  Focus: {info['focus']}\n"
                identity_text += "\n"
        documents.append(identity_text)
        metadatas.append({"source": "student_organizations", "type": "identity_organizations"})
        ids.append("orgs_identity")
    
    # 23. FAQs
    faqs = orgs_data.get("faqs", [])
    for i, faq in enumerate(faqs):
        faq_text = f"""Student Organizations FAQ: {faq.get('question', '')}

        {faq.get('answer', '')}"""
        documents.append(faq_text)
        metadatas.append({
            "source": "student_organizations",
            "type": "faq",
            "question": faq.get("question", "")
        })
        ids.append(f"orgs_faq_{i+1}")
    
    return documents, metadatas, ids


"""
Faculty/Professors Embeddings for KU BabyJay RAG System
Converts faculty JSON data into documents for vector database
"""

import json
from typing import List, Dict, Any, Tuple


def prepare_faculty_documents(data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Convert faculty JSON data into documents for embedding.
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    doc_counter = 0
    
    # Overview document
    overview = data.get("overview", {})
    overview_text = f"""KU Faculty Directory Overview

{overview.get('description', '')}

Source: {overview.get('source', '')}
Last Updated: {overview.get('last_updated', '')}

Note: {overview.get('note', '')}

The University of Kansas has faculty across many colleges and departments including:
- School of Business (Capitol Federal Hall)
- Electrical Engineering and Computer Science - EECS (Eaton Hall)
- Department of English (Wescoe Hall)
- Department of Psychology (Fraser Hall)
- Department of Chemistry (Integrated Science Building)
- Department of Mathematics (Snow Hall)
- Department of Physics and Astronomy (Malott Hall)

To find specific faculty information, you can:
1. Visit the KU Online Directory at directory.ku.edu
2. Visit individual department websites
3. Contact department offices directly"""

    documents.append(overview_text)
    metadatas.append({
        "source": "faculty",
        "type": "overview"
    })
    ids.append(f"faculty_{doc_counter}")
    doc_counter += 1
    
    # Process each department
    for dept in data.get("departments", []):
        dept_name = dept.get("name", "")
        
        # Department overview document
        dept_overview = f"""{dept_name} - Department Overview

Building: {dept.get('building', '')}
Address: {dept.get('address', '')}
Phone: {dept.get('phone', '')}
Email: {dept.get('email', '')}
Website: {dept.get('website', '')}

"""
        # Add chair/dean info
        if dept.get('dean'):
            dept_overview += f"Dean: {dept.get('dean')}\n"
        if dept.get('chair'):
            dept_overview += f"Chair: {dept.get('chair')}\n"
        
        # Add areas/programs
        if dept.get('areas'):
            dept_overview += f"\nAcademic Areas: {', '.join(dept.get('areas', []))}\n"
        if dept.get('programs'):
            dept_overview += f"\nPrograms: {', '.join(dept.get('programs', []))}\n"
        if dept.get('research_areas'):
            dept_overview += f"\nResearch Areas: {', '.join(dept.get('research_areas', []))}\n"
        
        # Add faculty count
        faculty_list = dept.get('faculty', [])
        dept_overview += f"\nNumber of Faculty: {len(faculty_list)}"
        
        documents.append(dept_overview)
        metadatas.append({
            "source": "faculty",
            "type": "department_overview",
            "department": dept_name
        })
        ids.append(f"faculty_{doc_counter}")
        doc_counter += 1
        
        # Create document with all faculty names for the department (for searching)
        faculty_names = [f.get('name', '') for f in faculty_list]
        faculty_list_doc = f"""{dept_name} - Faculty List

The following professors and instructors are in the {dept_name}:

{chr(10).join(['- ' + name for name in faculty_names])}

To find more information about a specific professor, visit {dept.get('website', '')} or contact the department at {dept.get('email', '')}.
"""
        documents.append(faculty_list_doc)
        metadatas.append({
            "source": "faculty",
            "type": "faculty_list",
            "department": dept_name
        })
        ids.append(f"faculty_{doc_counter}")
        doc_counter += 1
        
        # Create individual faculty documents (group by 5 for efficiency)
        faculty_groups = [faculty_list[i:i+5] for i in range(0, len(faculty_list), 5)]
        
        for group_idx, group in enumerate(faculty_groups):
            group_doc = f"{dept_name} - Faculty Profiles (Group {group_idx + 1})\n\n"
            
            for faculty in group:
                name = faculty.get('name', '')
                title = faculty.get('title', '')
                
                group_doc += f"**{name}**\n"
                group_doc += f"Title: {title}\n"
                
                # Add area/program/research based on what's available
                if faculty.get('area'):
                    group_doc += f"Area: {faculty.get('area')}\n"
                if faculty.get('program'):
                    group_doc += f"Program: {faculty.get('program')}\n"
                if faculty.get('research'):
                    group_doc += f"Research: {faculty.get('research')}\n"
                if faculty.get('email'):
                    group_doc += f"Email: {faculty.get('email')}\n"
                
                group_doc += "\n"
            
            group_doc += f"Department: {dept_name}\n"
            group_doc += f"Department Website: {dept.get('website', '')}\n"
            group_doc += f"Department Email: {dept.get('email', '')}"
            
            documents.append(group_doc)
            metadatas.append({
                "source": "faculty",
                "type": "faculty_profiles",
                "department": dept_name,
                "group": group_idx + 1
            })
            ids.append(f"faculty_{doc_counter}")
            doc_counter += 1
        
        # Create documents for leadership positions
        leaders = [f for f in faculty_list if any(term in f.get('title', '').lower() for term in 
                  ['dean', 'chair', 'director', 'distinguished', 'associate dean'])]
        
        if leaders:
            leadership_doc = f"{dept_name} - Leadership and Distinguished Faculty\n\n"
            
            for leader in leaders:
                leadership_doc += f"**{leader.get('name', '')}**\n"
                leadership_doc += f"Title: {leader.get('title', '')}\n"
                if leader.get('area'):
                    leadership_doc += f"Area: {leader.get('area')}\n"
                if leader.get('program'):
                    leadership_doc += f"Program: {leader.get('program')}\n"
                if leader.get('research'):
                    leadership_doc += f"Research: {leader.get('research')}\n"
                leadership_doc += "\n"
            
            leadership_doc += f"Department Contact: {dept.get('email', '')}"
            
            documents.append(leadership_doc)
            metadatas.append({
                "source": "faculty",
                "type": "department_leadership",
                "department": dept_name
            })
            ids.append(f"faculty_{doc_counter}")
            doc_counter += 1
    
    # Process FAQs
    for faq in data.get("faqs", []):
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        
        faq_doc = f"""Faculty FAQ: {question}

{answer}"""
        
        documents.append(faq_doc)
        metadatas.append({
            "source": "faculty",
            "type": "faq",
            "question": question
        })
        ids.append(f"faculty_{doc_counter}")
        doc_counter += 1
    
    print(f"Generated {len(documents)} faculty documents")
    return documents, metadatas, ids


def get_department_faculty(data: Dict[str, Any], department_name: str) -> List[Dict[str, Any]]:
    """Get all faculty for a specific department."""
    for dept in data.get("departments", []):
        if department_name.lower() in dept.get("name", "").lower():
            return dept.get("faculty", [])
    return []


def search_faculty_by_name(data: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    """Search for faculty by name across all departments."""
    results = []
    name_lower = name.lower()
    
    for dept in data.get("departments", []):
        for faculty in dept.get("faculty", []):
            if name_lower in faculty.get("name", "").lower():
                result = faculty.copy()
                result["department"] = dept.get("name")
                result["department_email"] = dept.get("email")
                result["department_phone"] = dept.get("phone")
                result["department_building"] = dept.get("building")
                results.append(result)
    
    return results


def search_faculty_by_research(data: Dict[str, Any], research_area: str) -> List[Dict[str, Any]]:
    """Search for faculty by research area."""
    results = []
    area_lower = research_area.lower()
    
    for dept in data.get("departments", []):
        for faculty in dept.get("faculty", []):
            research = faculty.get("research", "") or faculty.get("area", "") or faculty.get("program", "")
            if area_lower in research.lower():
                result = faculty.copy()
                result["department"] = dept.get("name")
                results.append(result)
    
    return results


# Test the module
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "overview": {
            "description": "KU Faculty Directory",
            "source": "Official KU websites",
            "last_updated": "December 2024",
            "note": "Visit department websites for most current information"
        },
        "departments": [
            {
                "name": "Test Department",
                "building": "Test Hall",
                "address": "123 Test St",
                "phone": "785-555-1234",
                "email": "test@ku.edu",
                "website": "test.ku.edu",
                "chair": "Dr. Test Chair",
                "faculty": [
                    {"name": "John Smith", "title": "Professor", "research": "Testing"},
                    {"name": "Jane Doe", "title": "Associate Professor", "research": "Research Methods"}
                ]
            }
        ],
        "faqs": [
            {
                "question": "How do I contact a professor?",
                "answer": "Email them or visit during office hours."
            }
        ]
    }
    
    docs, metas, doc_ids = prepare_faculty_documents(sample_data)
    
    print(f"\nGenerated {len(docs)} documents")
    print("\nSample document:")
    print(docs[0][:500] + "...")
    print("\nSample metadata:")
    print(metas[0])



# ============== END NEW FUNCTIONS ==============
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################



def initialize_database(persist_directory: str = None) -> chromadb.Collection:
    """
    Initialize ChromaDB and load all data.
    
    Args:
        persist_directory: Where to store the database. If None, uses default.
    
    Returns:
        ChromaDB collection with all embedded data
    """
    project_root = get_project_root()
    
    if persist_directory is None:
        persist_directory = str(project_root / "data" / "vectordb")
    
    # Create ChromaDB client with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Use OpenAI embedding function
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="babyjay_knowledge",
        embedding_function=embedding_fn,
        metadata={"description": "KU campus information for BabyJay chatbot"}
    )
    
    # Check if already populated
    if collection.count() > 0:
        print(f"Database already contains {collection.count()} documents")
        return collection
    
    print("Initializing database with campus data...")
    
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    # Load dining data
    dining_path = project_root / "data" / "dining" / "locations.json"
    if dining_path.exists():
        print(f"  Loading dining data from {dining_path}")
        dining_data = load_json_file(str(dining_path))
        docs, metas, ids = prepare_dining_documents(dining_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} dining documents")
    else:
        print(f"  Warning: Dining data not found at {dining_path}")
    
    # Load transit data
    transit_path = project_root / "data" / "transit" / "routes.json"
    if transit_path.exists():
        print(f"  Loading transit data from {transit_path}")
        transit_data = load_json_file(str(transit_path))
        docs, metas, ids = prepare_transit_documents(transit_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} transit documents")
    else:
        print(f"  Warning: Transit data not found at {transit_path}")
    
    # Load course data
    course_path = project_root / "data" / "courses" / "catalog.json"
    if course_path.exists():
        print(f"  Loading course data from {course_path}")
        course_data = load_json_file(str(course_path))
        docs, metas, ids = prepare_course_documents(course_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} course documents")
    else:
        print(f"  Warning: Course data not found at {course_path}")

    # Load building data
    building_path = project_root / "data" / "buildings" / "buildings.json"
    if building_path.exists():
        print(f"  Loading building data from {building_path}")
        building_data = load_json_file(str(building_path))
        docs, metas, ids = prepare_building_documents(building_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} building documents")
    else:
        print(f"  Warning: Building data not found at {building_path}")
    
    # Load office data
    office_path = project_root / "data" / "offices" / "offices.json"
    if office_path.exists():
        print(f"  Loading office data from {office_path}")
        office_data = load_json_file(str(office_path))
        docs, metas, ids = prepare_office_documents(office_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} office documents")
    else:
        print(f"  Warning: Office data not found at {office_path}")
    
    # Load professor data
    professor_path = project_root / "data" / "professors" / "professors.json"
    if professor_path.exists():
        print(f"  Loading professor data from {professor_path}")
        professor_data = load_json_file(str(professor_path))
        docs, metas, ids = prepare_professor_documents(professor_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} professor documents")
    else:
        print(f"  Warning: Professor data not found at {professor_path}")

    
    # Load admission data
    admission_path = project_root / "data" / "admissions" / "admissions.json"
    if admission_path.exists():
        print(f"  Loading admission data from {admission_path}")
        admission_data = load_json_file(str(admission_path))
        docs, metas, ids = prepare_admission_documents(admission_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} admission documents")
    else:
        print(f"  Warning: Admission data not found at {admission_path}")
    
    # Load academic calendar data
    calendar_path = project_root / "data" / "academic_calendar" / "academic_calendar.json"
    if calendar_path.exists():
        print(f"  Loading calendar data from {calendar_path}")
        calendar_data = load_json_file(str(calendar_path))
        docs, metas, ids = prepare_calendar_documents(calendar_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} calendar documents")
    else:
        print(f"  Warning: Calendar data not found at {calendar_path}")
    
    # Load FAQ data
    faq_path = project_root / "data" / "faqs" / "faqs.json"
    if faq_path.exists():
        print(f"  Loading FAQ data from {faq_path}")
        faq_data = load_json_file(str(faq_path))
        docs, metas, ids = prepare_faq_documents(faq_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} FAQ documents")
    else:
        print(f"  Warning: FAQ data not found at {faq_path}")

    # Load tuition data
    tuition_path = project_root / "data" / "tuition" / "tuition_fees.json"
    if tuition_path.exists():
        print(f"  Loading tuition data from {tuition_path}")
        tuition_data = load_json_file(str(tuition_path))
        docs, metas, ids = prepare_tuition_documents(tuition_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} tuition documents")
    else:
        print(f"  Warning: Tuition data not found at {tuition_path}")
    
    # Load financial aid data
    finaid_path = project_root / "data" / "financial_aid" / "financial_aid.json"
    if finaid_path.exists():
        print(f"  Loading financial aid data from {finaid_path}")
        finaid_data = load_json_file(str(finaid_path))
        docs, metas, ids = prepare_financial_aid_documents(finaid_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} financial aid documents")
    else:
        print(f"  Warning: Financial aid data not found at {finaid_path}")
    
    # Load housing data
    housing_path = project_root / "data" / "housing" / "housing.json"
    if housing_path.exists():
        print(f"  Loading housing data from {housing_path}")
        housing_data = load_json_file(str(housing_path))
        docs, metas, ids = prepare_housing_documents(housing_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"    Added {len(docs)} housing documents")
    else:
        print(f"  Warning: Housing data not found at {housing_path}")


    # Load library data
    library_path = project_root / "data" / "libraries" / "libraries.json"
    if library_path.exists():
        with open(library_path, 'r') as f:
            library_data = json.load(f)
        
        docs, metas, ids = prepare_library_documents(library_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"Loaded {len(docs)} library documents")

    # Load recreation data
    recreation_path = project_root / "data" / "recreation" / "recreation.json"
    if recreation_path.exists():
        with open(recreation_path, 'r') as f:
            recreation_data = json.load(f)
        docs, metas, ids = prepare_recreation_documents(recreation_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"Loaded {len(docs)} recreation documents")

    
    # Load campus safety data
    safety_path = project_root / "data" / "campus_safety" / "campus_safety.json"
    if safety_path.exists():
        with open(safety_path, 'r') as f:
            safety_data = json.load(f)
        docs, metas, ids = prepare_campus_safety_documents(safety_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"Loaded {len(docs)} campus safety documents")

    # Load student organizations data
    orgs_path = project_root / "data" / "student_organizations" / "student_organizations.json"
    if orgs_path.exists():
        with open(orgs_path, 'r') as f:
            orgs_data = json.load(f)
            docs, metas, ids = prepare_student_orgs_documents(orgs_data)
            all_documents.extend(docs)
            all_metadatas.extend(metas)
            all_ids.extend(ids)
            print(f"Loaded {len(docs)} student organization documents")

    # Faculty
    faculty_path = project_root / "data" / "faculty_data" / "faculty_data.json"
    if faculty_path.exists():
        with open(faculty_path, 'r') as f:
            faculty_data = json.load(f)
        docs, metas, ids = prepare_faculty_documents(faculty_data)
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        all_ids.extend(ids)
        print(f"Loaded {len(docs)} faculty documents")

    
# ====================================== END NEW DATA LOADING ===========================================
        
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
    
    # Add all documents to collection
    if all_documents:
        print(f"\nAdding {len(all_documents)} total documents to ChromaDB...")
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            end_idx = min(i + batch_size, len(all_documents))
            collection.add(
                documents=all_documents[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_ids[i:end_idx]
            )
        
        print(f"Database initialized with {collection.count()} documents")
    else:
        print("⚠️ No data found to load!")
    
    return collection


def reset_database(persist_directory: str = None):
    """Delete and recreate the database"""
    project_root = get_project_root()
    
    if persist_directory is None:
        persist_directory = str(project_root / "data" / "vectordb")
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("babyjay_knowledge")
        print("Deleted existing collection")
    except Exception:
        pass
    
    # Reinitialize
    return initialize_database(persist_directory)


if __name__ == "__main__":
    # Test the initialization
    print("Testing database initialization...")
    collection = initialize_database()
    print(f"\nTotal documents in database: {collection.count()}")
    
    # Test a simple query
    results = collection.query(
        query_texts=["Where can I eat on campus?"],
        n_results=3
    )
    print("\nTest query: 'Where can I eat on campus?'")
    print("Top 3 results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n{i+1}. {doc[:200]}...")