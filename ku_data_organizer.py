"""
KU Data Organizer
==================
Takes scraped data from data/raw/ and organizes it into structured folders
for efficient retrieval by CourseRetriever and ProgramRetriever
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List

class DataOrganizer:
    """Organizes scraped KU data into structured format"""
    
    def __init__(self):
        # Find the most recent scraped files in data/raw/
        self.courses_file = self.find_latest_file('data/raw/courses_*.json')
        self.programs_file = self.find_latest_file('data/raw/programs_*.json')
        
        if not self.courses_file or not self.programs_file:
            raise FileNotFoundError("No scraped data found in data/raw/. Run scraper first!")
        
        print(f"Loading courses from: {self.courses_file}")
        print(f"Loading programs from: {self.programs_file}")
        
        # Load data
        with open(self.courses_file, 'r') as f:
            courses_data = json.load(f)
            self.courses = courses_data['courses']
        
        with open(self.programs_file, 'r') as f:
            programs_data = json.load(f)
            self.programs = programs_data['programs']
        
        print(f"\n✓ Loaded {len(self.courses)} courses")
        print(f"✓ Loaded {len(self.programs)} programs")
    
    def find_latest_file(self, pattern: str) -> str:
        """Find the most recent file matching pattern"""
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files)  # Most recent by timestamp in filename
    
    def organize_all(self):
        """Run all organization tasks"""
        print("\n" + "="*80)
        print("ORGANIZING DATA")
        print("="*80)
        
        # Create directory structure
        self.create_directories()
        
        # Organize courses
        print("\nOrganizing courses...")
        self.organize_courses_by_department()
        self.organize_courses_by_level()
        self.organize_courses_by_subject()
        self.save_all_courses()
        
        # Organize programs
        print("\nOrganizing programs...")
        self.organize_programs_by_type()
        self.organize_programs_by_school()
        self.save_all_programs()
        
        # Build relationships
        print("\nBuilding relationships...")
        self.build_prerequisite_map()
        self.build_program_requirements()
        
        print("\n" + "="*80)
        print("ORGANIZATION COMPLETE")
        print("="*80)
    
    def create_directories(self):
        """Create directory structure"""
        dirs = [
            "data/courses",
            "data/courses/by_department",
            "data/courses/by_level",
            "data/courses/by_subject",
            "data/programs",
            "data/programs/by_type",
            "data/programs/by_school",
            "data/relationships"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print("  ✓ Created directory structure")
    
    def organize_courses_by_department(self):
        """Group courses by department"""
        by_dept = defaultdict(list)
        
        for course in self.courses:
            dept = course.get('department', 'Unknown')
            by_dept[dept].append(course)
        
        # Save each department
        for dept, courses in by_dept.items():
            filename = dept.lower().replace(' & ', '_').replace(' ', '_').replace('/', '_')
            filepath = f"data/courses/by_department/{filename}.json"
            
            with open(filepath, 'w') as f:
                json.dump({
                    "department": dept,
                    "total_courses": len(courses),
                    "courses": courses
                }, f, indent=2)
        
        print(f"    ✓ Created {len(by_dept)} department files")
    
    def organize_courses_by_level(self):
        """Group courses by level"""
        by_level = defaultdict(list)
        
        for course in self.courses:
            level = course.get('level', 'unknown')
            by_level[level].append(course)
        
        # Save each level
        for level, courses in by_level.items():
            filepath = f"data/courses/by_level/{level}.json"
            
            with open(filepath, 'w') as f:
                json.dump({
                    "level": level,
                    "total_courses": len(courses),
                    "courses": courses
                }, f, indent=2)
        
        print(f"    ✓ Created {len(by_level)} level files")
    
    def organize_courses_by_subject(self):
        """Group courses by subject code"""
        by_subject = defaultdict(list)
        
        for course in self.courses:
            subject = course.get('subject', 'UNK')
            by_subject[subject].append(course)
        
        # Save each subject
        for subject, courses in by_subject.items():
            filepath = f"data/courses/by_subject/{subject.lower()}.json"
            
            with open(filepath, 'w') as f:
                json.dump({
                    "subject": subject,
                    "total_courses": len(courses),
                    "courses": courses
                }, f, indent=2)
        
        print(f"    ✓ Created {len(by_subject)} subject files")
    
    def save_all_courses(self):
        """Save master course file"""
        with open('data/courses/all_courses.json', 'w') as f:
            json.dump({
                "total": len(self.courses),
                "courses": self.courses
            }, f, indent=2)
        
        print(f"    ✓ Saved all_courses.json ({len(self.courses)} courses)")
    
    def organize_programs_by_type(self):
        """Group programs by type"""
        by_type = defaultdict(list)
        
        for prog_key, prog_data in self.programs.items():
            prog_type = prog_data.get('type', 'other')
            by_type[prog_type].append(prog_data)
        
        # Save each type
        for prog_type, programs in by_type.items():
            filepath = f"data/programs/by_type/{prog_type}.json"
            
            with open(filepath, 'w') as f:
                json.dump({
                    "type": prog_type,
                    "total_programs": len(programs),
                    "programs": programs
                }, f, indent=2)
        
        print(f"    ✓ Created {len(by_type)} program type files")
    
    def organize_programs_by_school(self):
        """Group programs by school"""
        by_school = defaultdict(list)
        
        for prog_key, prog_data in self.programs.items():
            school = prog_data.get('school', 'Unknown')
            by_school[school].append(prog_data)
        
        # Save each school
        for school, programs in by_school.items():
            filename = school.lower().replace(' & ', '_').replace(' ', '_')
            filepath = f"data/programs/by_school/{filename}.json"
            
            with open(filepath, 'w') as f:
                json.dump({
                    "school": school,
                    "total_programs": len(programs),
                    "programs": programs
                }, f, indent=2)
        
        print(f"    ✓ Created {len(by_school)} school program files")
    
    def save_all_programs(self):
        """Save master program file"""
        with open('data/programs/all_programs.json', 'w') as f:
            json.dump({
                "total": len(self.programs),
                "programs": self.programs
            }, f, indent=2)
        
        print(f"    ✓ Saved all_programs.json ({len(self.programs)} programs)")
    
    def build_prerequisite_map(self):
        """Build prerequisite relationship map"""
        prereq_map = {}
        
        for course in self.courses:
            code = course.get('course_code')
            prereqs_text = course.get('prerequisites')
            
            if prereqs_text:
                # Simple extraction - find course codes in prerequisites
                prereq_codes = self.extract_course_codes(prereqs_text)
                if prereq_codes:
                    prereq_map[code] = prereq_codes
        
        # Save
        filepath = "data/relationships/prerequisites.json"
        with open(filepath, 'w') as f:
            json.dump(prereq_map, f, indent=2)
        
        print(f"    ✓ Built prerequisite map ({len(prereq_map)} courses with prereqs)")
    
    def build_program_requirements(self):
        """Build program → required courses map"""
        prog_requirements = {}
        
        for prog_key, prog_data in self.programs.items():
            prog_requirements[prog_key] = {
                "school": prog_data['school'],
                "name": prog_data['name'],
                "type": prog_data['type'],
                "required_courses": prog_data.get('required_courses', []),
                "degree_plan": prog_data.get('degree_plan', []),
                "admission_requirements": prog_data.get('admission_requirements', '')[:500],  # Truncate
                "learning_outcomes": prog_data.get('learning_outcomes', [])
            }
        
        # Save
        filepath = "data/relationships/program_requirements.json"
        with open(filepath, 'w') as f:
            json.dump(prog_requirements, f, indent=2)
        
        print(f"    ✓ Built program requirements ({len(prog_requirements)} programs)")
    
    def extract_course_codes(self, text: str) -> List[str]:
        """Extract course codes from prerequisite text"""
        import re
        
        # Find patterns like "ARCH 100", "EECS 168"
        pattern = r'\b([A-Z]{2,5})\s+(\d{3})\b'
        matches = re.findall(pattern, text)
        
        # Combine subject and number
        codes = [f"{subj} {num}" for subj, num in matches]
        
        return list(set(codes))  # Remove duplicates
    
    def print_summary(self):
        """Print organization summary"""
        print("\n" + "="*80)
        print("FINAL STRUCTURE")
        print("="*80)
        print("\ndata/")
        print("  raw/                       (original scraped data)")
        print("    courses_TIMESTAMP.json")
        print("    programs_TIMESTAMP.json")
        print("    stats_TIMESTAMP.json")
        print("")
        print("  courses/")
        print("    by_department/           (organized by department)")
        print("    by_level/                (undergraduate/graduate)")
        print("    by_subject/              (ARCH, EECS, MATH, etc.)")
        print("    all_courses.json         (master file)")
        print("")
        print("  programs/")
        print("    by_type/                 (bachelors/masters/doctoral)")
        print("    by_school/               (organized by school)")
        print("    all_programs.json        (master file)")
        print("")
        print("  relationships/")
        print("    prerequisites.json       (course dependencies)")
        print("    program_requirements.json (program course lists)")
        print("")
        print("="*80)
        print("✓ Data is ready for CourseRetriever and ProgramRetriever!")
        print("="*80)


def main():
    """Run the organizer"""
    try:
        organizer = DataOrganizer()
        organizer.organize_all()
        organizer.print_summary()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you've run the scraper first:")
        print("  python ku_comprehensive_scraper.py ku_urls.json")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()