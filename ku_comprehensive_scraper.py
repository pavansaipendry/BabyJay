"""
KU Comprehensive Catalog Scraper - FIXED VERSION
==================================
Improved course extraction with multiple parsing strategies
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from datetime import datetime
from typing import List, Dict, Optional

REQUEST_DELAY = 2  # seconds


class KUComprehensiveScraper:
    """Scrapes KU catalog from provided URL list"""
    
    def __init__(self, urls_file: str):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KU BabyJay Educational Scraper (student project)'
        })
        
        # Load URLs
        with open(urls_file, 'r') as f:
            self.url_structure = json.load(f)
        
        # Storage
        self.courses = []
        self.program_details = {}
        self.stats = {
            "urls_scraped": 0,
            "courses_found": 0,
            "programs_found": 0,
            "errors": []
        }
    
    def scrape_all(self):
        """Scrape everything"""
        print("=" * 80)
        print("KU COMPREHENSIVE SCRAPER - FIXED VERSION")
        print("=" * 80)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        for school_name, school_data in self.url_structure.items():
            print(f"\n{'='*80}")
            print(f"SCHOOL: {school_name}")
            print(f"{'='*80}")
            
            # Scrape departments
            for dept in school_data.get("departments", []):
                self.scrape_department(school_name, dept)
            
            # Scrape programs
            for prog in school_data.get("programs", []):
                self.scrape_program(school_name, prog)
        
        print("\n" + "=" * 80)
        print("SCRAPING COMPLETE")
        print("=" * 80)
        self.print_stats()
    
    def scrape_department(self, school: str, dept: Dict):
        """Scrape a department's URLs"""
        dept_name = dept["name"]
        print(f"\n  DEPARTMENT: {dept_name}")
        
        for url in dept["urls"]:
            self.scrape_url(url, school, dept_name, "department")
            time.sleep(REQUEST_DELAY)
    
    def scrape_program(self, school: str, prog: Dict):
        """Scrape a program's URLs"""
        prog_name = prog["name"]
        print(f"\n  PROGRAM: {prog_name}")
        
        # Initialize program details
        program_key = f"{school}::{prog_name}"
        self.program_details[program_key] = {
            "school": school,
            "name": prog_name,
            "type": self.determine_program_type(prog_name),
            "admission_requirements": "",
            "degree_requirements": "",
            "degree_plan": [],
            "learning_outcomes": [],
            "required_courses": [],
            "urls": prog["urls"]
        }
        
        for url in prog["urls"]:
            self.scrape_url(url, school, prog_name, "program")
            time.sleep(REQUEST_DELAY)
    
    def scrape_url(self, url: str, school: str, entity_name: str, entity_type: str):
        """Scrape a single URL"""
        try:
            print(f"    → {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Determine what type of content this is
            if "#courseinventory" in url:
                self.extract_courses(soup, school, entity_name, url)
            elif "#degreerequirementstext" in url or "#requirementstext" in url or "#certificaterequirementstext" in url or "#minorrequirementstext" in url:
                self.extract_requirements(soup, school, entity_name)
            elif "#degreeplantext" in url or "#plantext" in url or "#graduationplantext" in url:
                self.extract_degree_plan(soup, school, entity_name)
            elif "#admissiontext" in url or "#admissionstext" in url:
                self.extract_admission_info(soup, school, entity_name)
            elif "#learningoutcomestext" in url:
                self.extract_learning_outcomes(soup, school, entity_name)
            else:
                # Generic extraction
                self.extract_generic(soup, school, entity_name, url)
            
            self.stats["urls_scraped"] += 1
            
        except Exception as e:
            error_msg = f"Error scraping {url}: {str(e)}"
            print(f"      ✗ {error_msg}")
            self.stats["errors"].append(error_msg)
    
    def extract_courses(self, soup: BeautifulSoup, school: str, entity_name: str, url: str):
        """
        IMPROVED course extraction with multiple strategies
        """
        courses_found = 0
        
        # Get the full text to analyze
        full_text = soup.get_text()
        
        # Find all potential course codes in the text first
        course_code_pattern = r'\b([A-Z]{2,5})\s+(\d{3})\b'
        potential_courses = re.findall(course_code_pattern, full_text)
        
        if potential_courses:
            print(f"      → Found {len(set(potential_courses))} potential course codes, parsing details...")
        
        # STRATEGY 1: Look for courseblock class (common in course catalogs)
        course_blocks = soup.find_all(class_=re.compile(r'courseblock'))
        if course_blocks:
            print(f"      → Found {len(course_blocks)} courseblock elements")
            for block in course_blocks:
                course = self.parse_course_block(block, school, entity_name)
                if course:
                    self.courses.append(course)
                    courses_found += 1
        
        # STRATEGY 2: Look for div.course or div.courseDescription
        course_divs = soup.find_all('div', class_=re.compile(r'course'))
        if course_divs:
            print(f"      → Found {len(course_divs)} course div elements")
            for div in course_divs:
                course = self.parse_course_block(div, school, entity_name)
                if course:
                    self.courses.append(course)
                    courses_found += 1
        
        # STRATEGY 3: Text-based parsing - split by course codes
        if courses_found == 0 and potential_courses:
            print(f"      → Trying text-based parsing...")
            courses_found = self.parse_courses_from_text(full_text, school, entity_name)
        
        # STRATEGY 4: Look for <p> tags with strong course codes
        if courses_found == 0:
            for p in soup.find_all('p'):
                strong = p.find('strong')
                if not strong:
                    continue
                
                strong_text = strong.get_text().strip()
                if re.match(r'^[A-Z]{2,5}\s+\d{3}', strong_text):
                    course = self.parse_course_paragraph(p, school, entity_name)
                    if course:
                        self.courses.append(course)
                        courses_found += 1
        
        if courses_found > 0:
            print(f"      ✓ Extracted {courses_found} courses")
            self.stats["courses_found"] += courses_found
        else:
            print(f"      ⚠ No courses extracted (found {len(set(potential_courses))} course codes in text)")
    
    def parse_course_block(self, element, school: str, department: str) -> Optional[Dict]:
        """Parse a course from a structured HTML element"""
        text = element.get_text()
        
        # Extract course code
        code_match = re.search(r'([A-Z]{2,5})\s+(\d{3})', text)
        if not code_match:
            return None
        
        subject = code_match.group(1)
        number = code_match.group(2)
        course_code = f"{subject} {number}"
        
        # Extract title and credits - try multiple patterns
        title_match = re.search(r'([A-Z]{2,5}\s+\d{3})\.?\s+(.+?)\.\s+(\d+)\s+[Cc]redits?', text)
        if not title_match:
            # Try without period after course code
            title_match = re.search(r'([A-Z]{2,5}\s+\d{3})\s+(.+?)\s+(\d+)\s+[Cc]redits?', text)
        
        if not title_match:
            return None
        
        title = title_match.group(2).strip()
        try:
            credits = int(title_match.group(3))
        except:
            credits = 0
        
        # Extract description
        desc_match = re.search(r'\d+\s+[Cc]redits?\.\s+(.+?)(?:Prerequisite:|Corequisite:|$)', text, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Extract prerequisites
        prereq_match = re.search(r'Prerequisite:\s+(.+?)(?:\.|Corequisite:|$)', text, re.DOTALL)
        prerequisites = prereq_match.group(1).strip() if prereq_match else None
        
        # Extract corequisites
        coreq_match = re.search(r'Corequisite:\s+(.+?)(?:\.|$)', text, re.DOTALL)
        corequisites = coreq_match.group(1).strip() if coreq_match else None
        
        # Determine level
        level = "graduate" if int(number) >= 500 else "undergraduate"
        
        # Check for KU Core
        ku_core = None
        core_patterns = [r'GE\d+', r'AE\d+', r'\b(AH|NPS|NLEC|NLAB|SBS|USC|CAP|MTS|CMS|ENG)\b']
        for pattern in core_patterns:
            match = re.search(pattern, text)
            if match:
                ku_core = match.group(0)
                break
        
        return {
            "subject": subject,
            "number": number,
            "course_code": course_code,
            "title": title,
            "credits": credits,
            "level": level,
            "description": description,
            "prerequisites": prerequisites,
            "corequisites": corequisites,
            "school": school,
            "department": department,
            "ku_core": ku_core,
            "scraped_at": datetime.now().isoformat()
        }
    
    def parse_course_paragraph(self, p_element, school: str, department: str) -> Optional[Dict]:
        """Parse course from a <p> tag"""
        return self.parse_course_block(p_element, school, department)
    
    def parse_courses_from_text(self, text: str, school: str, department: str) -> int:
        """Parse courses directly from text by splitting on course codes"""
        courses_found = 0
        
        # Split text into sections by course codes
        course_pattern = r'([A-Z]{2,5}\s+\d{3})'
        parts = re.split(course_pattern, text)
        
        # Process pairs (code, description)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                course_code = parts[i]
                course_text = parts[i] + " " + parts[i+1]
                
                # Only process if it looks like a course description
                if 'credit' in course_text.lower() and len(course_text) > 50:
                    course = self.parse_course_block_from_text(course_text, school, department)
                    if course:
                        self.courses.append(course)
                        courses_found += 1
        
        return courses_found
    
    def parse_course_block_from_text(self, text: str, school: str, department: str) -> Optional[Dict]:
        """Parse course from plain text"""
        # Use same logic as parse_course_block but on plain text
        code_match = re.search(r'([A-Z]{2,5})\s+(\d{3})', text)
        if not code_match:
            return None
        
        subject = code_match.group(1)
        number = code_match.group(2)
        course_code = f"{subject} {number}"
        
        # Extract title and credits
        title_match = re.search(r'([A-Z]{2,5}\s+\d{3})\.?\s+(.+?)\.\s+(\d+(?:-\d+)?)\s+[Cc]redits?', text)
        if not title_match:
            return None
        
        title = title_match.group(2).strip()
        try:
            credits_str = title_match.group(3)
            if '-' in credits_str:
                credits = int(credits_str.split('-')[0])  # Take lower bound
            else:
                credits = int(credits_str)
        except:
            credits = 0
        
        # Extract description
        desc_match = re.search(r'\d+(?:-\d+)?\s+[Cc]redits?\.\s+(.+?)(?:Prerequisite:|Corequisite:|LEC|$)', text, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Clean up description
        description = ' '.join(description.split())  # Normalize whitespace
        if len(description) > 500:
            description = description[:500]
        
        # Prerequisites
        prereq_match = re.search(r'Prerequisite:\s+(.+?)(?:\.|Corequisite:|$)', text, re.DOTALL)
        prerequisites = prereq_match.group(1).strip() if prereq_match else None
        
        # Corequisites
        coreq_match = re.search(r'Corequisite:\s+(.+?)(?:\.|$)', text, re.DOTALL)
        corequisites = coreq_match.group(1).strip() if coreq_match else None
        
        # Level
        level = "graduate" if int(number) >= 500 else "undergraduate"
        
        # KU Core
        ku_core = None
        core_patterns = [r'GE\d+', r'AE\d+', r'\b(AH|NPS|NLEC|NLAB|SBS|USC|CAP|MTS|CMS|ENG)\b']
        for pattern in core_patterns:
            match = re.search(pattern, text)
            if match:
                ku_core = match.group(0)
                break
        
        return {
            "subject": subject,
            "number": number,
            "course_code": course_code,
            "title": title,
            "credits": credits,
            "level": level,
            "description": description,
            "prerequisites": prerequisites,
            "corequisites": corequisites,
            "school": school,
            "department": department,
            "ku_core": ku_core,
            "scraped_at": datetime.now().isoformat()
        }
    
    def extract_requirements(self, soup: BeautifulSoup, school: str, prog_name: str):
        """Extract degree/program requirements"""
        program_key = f"{school}::{prog_name}"
        
        content = soup.get_text()
        course_codes = self.extract_course_codes(content)
        
        if program_key in self.program_details:
            self.program_details[program_key]["degree_requirements"] = content[:2000]
            self.program_details[program_key]["required_courses"].extend(course_codes)
            
            self.program_details[program_key]["required_courses"] = list(set(
                self.program_details[program_key]["required_courses"]
            ))
            
            print(f"      ✓ Found {len(course_codes)} required courses")
    
    def extract_degree_plan(self, soup: BeautifulSoup, school: str, prog_name: str):
        """Extract degree plan (course sequence)"""
        program_key = f"{school}::{prog_name}"
        
        tables = soup.find_all('table')
        
        plan = []
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_text = ' '.join([cell.get_text().strip() for cell in cells])
                
                codes = self.extract_course_codes(row_text)
                if codes:
                    plan.extend(codes)
        
        if program_key in self.program_details:
            self.program_details[program_key]["degree_plan"] = plan
            print(f"      ✓ Degree plan with {len(plan)} courses")
    
    def extract_admission_info(self, soup: BeautifulSoup, school: str, prog_name: str):
        """Extract admission requirements"""
        program_key = f"{school}::{prog_name}"
        
        content = soup.get_text()
        
        if program_key in self.program_details:
            self.program_details[program_key]["admission_requirements"] = content[:2000]
            print(f"      ✓ Admission requirements")
    
    def extract_learning_outcomes(self, soup: BeautifulSoup, school: str, prog_name: str):
        """Extract learning outcomes"""
        program_key = f"{school}::{prog_name}"
        
        outcomes = []
        for li in soup.find_all('li'):
            text = li.get_text().strip()
            if len(text) > 20:
                outcomes.append(text)
        
        if program_key in self.program_details:
            self.program_details[program_key]["learning_outcomes"] = outcomes
            print(f"      ✓ {len(outcomes)} learning outcomes")
    
    def extract_generic(self, soup: BeautifulSoup, school: str, entity_name: str, url: str):
        """Generic extraction for other page types"""
        text = soup.get_text()
        course_codes = self.extract_course_codes(text)
        
        if course_codes:
            print(f"      ✓ Found {len(course_codes)} course references")
    
    def extract_course_codes(self, text: str) -> List[str]:
        """Extract all course codes from text"""
        pattern = r'\b([A-Z]{2,5})\s+(\d{3})\b'
        matches = re.findall(pattern, text)
        
        codes = [f"{subj} {num}" for subj, num in matches]
        return list(set(codes))
    
    def determine_program_type(self, name: str) -> str:
        """Determine program type from name"""
        name_lower = name.lower()
        
        if 'ph.d' in name_lower or 'doctor' in name_lower:
            return "doctoral"
        elif 'master' in name_lower or 'm.a.' in name_lower or 'm.s.' in name_lower or 'm.arch' in name_lower:
            return "masters"
        elif 'bachelor' in name_lower or 'b.a.' in name_lower or 'b.s.' in name_lower or 'b.f.a.' in name_lower:
            return "bachelors"
        elif 'certificate' in name_lower:
            return "certificate"
        elif 'minor' in name_lower:
            return "minor"
        else:
            return "other"
    
    def save_results(self):
        """Save all scraped data to data/raw/"""
        os.makedirs('data/raw', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save courses
        courses_file = f"data/raw/courses_{timestamp}.json"
        with open(courses_file, 'w', encoding='utf-8') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "data_source": "catalog.ku.edu (comprehensive)",
                "total_courses": len(self.courses),
                "courses": self.courses
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Courses saved: {courses_file}")
        
        # Save program details
        programs_file = f"data/raw/programs_{timestamp}.json"
        with open(programs_file, 'w', encoding='utf-8') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "data_source": "catalog.ku.edu (comprehensive)",
                "total_programs": len(self.program_details),
                "programs": self.program_details
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Programs saved: {programs_file}")
        
        # Save statistics
        stats_file = f"data/raw/stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✓ Statistics saved: {stats_file}")
    
    def print_stats(self):
        """Print statistics"""
        print(f"\nURLs scraped: {self.stats['urls_scraped']}")
        print(f"Courses found: {self.stats['courses_found']}")
        print(f"Programs found: {len(self.program_details)}")
        
        if self.stats['errors']:
            print(f"\nErrors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")


def main():
    """Run scraper"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ku_comprehensive_scraper.py <urls.json>")
        print("\nExample:")
        print("  python ku_comprehensive_scraper.py ku_urls.json")
        sys.exit(1)
    
    urls_file = sys.argv[1]
    
    if not os.path.exists(urls_file):
        print(f"Error: {urls_file} not found")
        sys.exit(1)
    
    scraper = KUComprehensiveScraper(urls_file)
    
    try:
        scraper.scrape_all()
        scraper.save_results()
        
        print("\n" + "=" * 80)
        print("SUCCESS! Data saved to data/raw/")
        print("=" * 80)
        print("\nNext step: Organize the data")
        print("  python ku_data_organizer.py")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Saving partial results...")
        scraper.save_results()
    
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Saving partial results...")
        scraper.save_results()


if __name__ == "__main__":
    main()