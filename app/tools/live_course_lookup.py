"""
Live Course Lookup for BabyJay
==============================
Queries classes.ku.edu in real-time for current course info.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List
import re


# Semester codes
SEMESTER_CODES = {
    "Spring 2026": "4262",
    "Fall 2025": "4258",
    "Summer 2025": "4254",
    "Spring 2025": "4252",
}

CURRENT_SEMESTER = "Spring 2026"
CURRENT_TERM_CODE = SEMESTER_CODES[CURRENT_SEMESTER]


def lookup_course(
    query: str,
    career: str = "Graduate",
    semester: str = None,
    timeout: float = 10.0
) -> Dict:
    """Look up course info from classes.ku.edu in real-time."""
    
    semester = semester or CURRENT_SEMESTER
    term_code = SEMESTER_CODES.get(semester, CURRENT_TERM_CODE)
    
    form_data = {
        "classesSearchText": query,
        "searchCareer": career,
        "searchTerm": term_code,
        "searchSchool": "",
        "searchDept": "",
        "searchSubject": "",
        "searchCode": "",
        "textbookOptions": "",
        "searchCampus": "",
        "searchBuilding": "",
        "searchCourseNumberMin": "001",
        "searchCourseNumberMax": "999",
        "searchCreditHours": "",
        "searchInstructor": "",
        "searchStartTime": "",
        "searchEndTime": "",
        "searchClosed": "false",
        "searchHonorsClasses": "false",
        "searchShortClasses": "false",
        "searchOnlineClasses": "",
        "searchIncludeExcludeDays": "include",
        "searchDays": "",
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://classes.ku.edu",
        "Referer": "https://classes.ku.edu/",
    }
    
    try:
        response = requests.post(
            "https://classes.ku.edu/Classes/CourseSearch.action",
            data=form_data,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        
        sections = parse_course_html(response.text)
        
        return {
            "success": True,
            "query": query,
            "career": career,
            "semester": semester,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections": sections,
            "total_sections": len(sections),
            "source": "classes.ku.edu (live)"
        }
        
    except requests.Timeout:
        return {"success": False, "error": "Request timed out", "query": query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        return {"success": False, "error": str(e), "query": query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def parse_course_html(html: str) -> List[Dict]:
    """Parse the HTML response from classes.ku.edu."""
    
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    
    # Find all class_list tables directly
    class_tables = soup.find_all("table", class_="class_list")
    
    for class_table in class_tables:
        # Get course code from h3 (go up to find it)
        parent_table = class_table.find_parent("table")
        course_code = "Unknown"
        if parent_table:
            h3 = parent_table.find("h3")
            if h3:
                course_code = h3.get_text(strip=True)
        
        # Parse rows - they come in pairs (main row + notes row)
        rows = class_table.find_all("tr")
        current_section = None
        
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            
            first_cell_text = cells[0].get_text(strip=True)
            
            # Main section row (LEC, LAB, etc.)
            if first_cell_text in ["LEC", "LAB", "DIS", "SEM", "IND", "RSC", "FLD"]:
                
                # Topic and instructor (cell 1)
                topic = ""
                instructor = "TBA"
                if len(cells) > 1:
                    cell_text = cells[1].get_text()
                    # Extract topic
                    if "Topic:" in cell_text:
                        # Get text between "Topic:" and the instructor link or end
                        topic_match = re.search(r"Topic:\s*(.+?)(?:\s{2,}|$)", cell_text)
                        if topic_match:
                            topic = topic_match.group(1).strip().replace('\xa0', ' ').strip()
                    
                    # Extract instructor from link
                    instructor_link = cells[1].find("a", href=re.compile(r"directory\.ku\.edu"))
                    if instructor_link:
                        instructor = instructor_link.get_text(strip=True)
                
                # Credits (cell 2)
                credits = ""
                if len(cells) > 2:
                    credits_text = cells[2].get_text(strip=True)
                    credits_match = re.match(r"(\d+)", credits_text)
                    if credits_match:
                        credits = credits_match.group(1)
                
                # Class number (cell 3)
                class_num = ""
                if len(cells) > 3:
                    strong = cells[3].find("strong")
                    if strong:
                        class_num = strong.get_text(strip=True)
                
                # Seats (cell 4)
                seats = "N/A"
                seats_status = "unknown"
                enrolled = None
                max_seats = None
                if len(cells) > 4:
                    span = cells[4].find("span")
                    if span:
                        seats_text = span.get_text(strip=True)
                        title = span.get("title", "")
                        
                        if "Unopened" in seats_text:
                            seats = "Unopened"
                            seats_status = "unopened"
                        elif seats_text.isdigit():
                            seats = int(seats_text)
                            seats_status = "open" if seats > 0 else "full"
                            # Parse "X students enrolled out of Y"
                            match = re.search(r"(\d+) students enrolled out of (\d+)", title)
                            if match:
                                enrolled = int(match.group(1))
                                max_seats = int(match.group(2))
                        elif "Closed" in seats_text or seats_text == "0":
                            seats = 0
                            seats_status = "full"
                
                current_section = {
                    "course": course_code,
                    "type": first_cell_text,
                    "topic": topic,
                    "instructor": instructor,
                    "credits": credits,
                    "class_number": class_num,
                    "seats": seats,
                    "seats_status": seats_status,
                    "enrolled": enrolled,
                    "max_seats": max_seats,
                    "days": "",
                    "time": "",
                    "location": "",
                }
                sections.append(current_section)
            
            # Notes row (contains time/location)
            elif first_cell_text == "Notes" and current_section:
                if len(cells) > 1:
                    time_cell = cells[1]
                    time_text = time_cell.get_text(" ", strip=True)
                    
                    # Parse days (MWF, TuTh, etc.)
                    days_match = re.match(r"((?:Mo|Tu|We|Th|Fr|Sa|Su|M|T|W|F)+)", time_text)
                    if days_match:
                        current_section["days"] = days_match.group(1)
                    
                    # Parse time
                    time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))\s*-\s*(\d{1,2}:\d{2}\s*(?:AM|PM))", time_text)
                    if time_match:
                        current_section["time"] = f"{time_match.group(1)} - {time_match.group(2)}"
                    
                    # Parse location
                    loc_link = time_cell.find("a", href=re.compile(r"maps\.google"))
                    if loc_link:
                        loc_span = loc_link.find("span")
                        current_section["location"] = loc_span.get_text(strip=True) if loc_span else loc_link.get_text(strip=True)
    
    return sections


def format_sections_for_chat(result: Dict) -> str:
    """Format the lookup result for chat display."""
    
    if not result.get("success"):
        return f"Sorry, I couldn't fetch live course data: {result.get('error', 'Unknown error')}"
    
    sections = result.get("sections", [])
    if not sections:
        return f"No sections found for '{result['query']}' in {result['semester']} ({result['career']} level)."
    
    timestamp = result.get("timestamp", "")
    lines = [
        f"**{result['query']}** - {result['semester']} ({result['career']})",
        f"_Live data as of {timestamp}_",
        "",
    ]
    
    for i, s in enumerate(sections, 1):
        emoji = "🟢" if s.get('seats_status') == 'open' else "🔴" if s.get('seats_status') == 'full' else "⚪"
        
        # Format seats with enrolled info
        if s.get('enrolled') is not None and s.get('max_seats') is not None:
            seats_str = f"{s['seats']} open ({s['enrolled']}/{s['max_seats']} enrolled)"
        else:
            seats_str = str(s.get('seats', 'N/A'))
        
        lines.append(f"**{i}. {s.get('topic') or 'Section'}**")
        lines.append(f"   Instructor: {s.get('instructor', 'TBA')}")
        lines.append(f"   Time: {s.get('days', '')} {s.get('time', 'TBA')}")
        lines.append(f"   Location: {s.get('location', 'TBA')}")
        lines.append(f"   {emoji} Seats: {seats_str}")
        lines.append(f"   Class #: {s.get('class_number', 'N/A')}")
        lines.append("")
    
    lines.append("_Seats can change instantly - enroll ASAP if interested!_")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Testing Live Course Lookup")
    print("=" * 60)
    
    print("\n🔍 Looking up EECS 700 (Graduate)...")
    result = lookup_course("EECS 700", career="Graduate")
    
    if result["success"]:
        print(f"Found {result['total_sections']} sections")
        print(f"Timestamp: {result['timestamp']}")
        print("\n" + format_sections_for_chat(result))
    else:
        print(f"Error: {result.get('error')}")
    
    print("\n" + "=" * 60)
    
    print("\n🔍 Looking up EECS 168 (Undergraduate)...")
    result = lookup_course("EECS 168", career="Undergraduate")
    
    if result["success"]:
        print(f"Found {result['total_sections']} sections")
        for s in result["sections"][:3]:
            print(f"   - {s.get('instructor')}: {s.get('days')} {s.get('time')} | Seats: {s.get('seats')}")
    else:
        print(f"Error: {result.get('error')}")