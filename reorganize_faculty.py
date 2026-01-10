#!/usr/bin/env python3
"""
Phase 1: Reorganize Faculty Data
================================
Splits all_faculty_combined.json into individual department files.

Creates:
  data/faculty/by_department/eecs.json
  data/faculty/by_department/physics.json
  data/faculty/by_department/business.json
  ... etc

Also creates:
  data/faculty/department_index.json (list of all departments with metadata)

Usage:
  cd BabyJay
  python reorganize_faculty.py
"""

import json
import os
from pathlib import Path

# Paths - adjust these if needed
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR  # Script should be run from BabyJay folder
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_FILE = DATA_DIR / "all_faculty_combined.json"
OUTPUT_DIR = DATA_DIR / "faculty" / "by_department"


def reorganize_faculty():
    """Split combined faculty JSON into per-department files."""
    
    # Check source file exists
    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        print("Make sure all_faculty_combined.json is in the data/ folder")
        return False
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load source data
    print(f"Loading {SOURCE_FILE}...")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Found {len(all_data)} departments")
    
    # Track department metadata
    department_index = []
    total_faculty = 0
    
    # Split into individual files
    for dept_key, dept_data in all_data.items():
        dept_name = dept_data.get('name', dept_key)
        faculty_list = dept_data.get('faculty', [])
        faculty_count = len(faculty_list)
        total_faculty += faculty_count
        
        # Add department info to each faculty member (useful for searches)
        for faculty in faculty_list:
            faculty['department'] = dept_name
            faculty['department_key'] = dept_key
        
        # Create department file with full info
        dept_file = {
            "department_key": dept_key,
            "department_name": dept_name,
            "building": dept_data.get('building', ''),
            "address": dept_data.get('address', ''),
            "phone": dept_data.get('phone', ''),
            "email": dept_data.get('email', ''),
            "website": dept_data.get('website', ''),
            "faculty_count": faculty_count,
            "faculty": faculty_list
        }
        
        # Save department file
        output_file = OUTPUT_DIR / f"{dept_key}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dept_file, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ {dept_key}.json ({faculty_count} faculty)")
        
        # Add to index
        department_index.append({
            "key": dept_key,
            "name": dept_name,
            "faculty_count": faculty_count,
            "file": f"by_department/{dept_key}.json"
        })
    
    # Save department index
    index_file = DATA_DIR / "faculty" / "department_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_departments": len(department_index),
            "total_faculty": total_faculty,
            "departments": sorted(department_index, key=lambda x: x['name'])
        }, f, indent=2)
    
    print(f"\n✓ Created department_index.json")
    print(f"\n{'='*50}")
    print(f"COMPLETE!")
    print(f"  Departments: {len(department_index)}")
    print(f"  Total Faculty: {total_faculty}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*50}")
    
    return True


def verify_output():
    """Verify the reorganization worked correctly."""
    print("\nVerifying output...")
    
    # Check a few department files
    test_depts = ['eecs', 'physics', 'business']
    
    for dept in test_depts:
        file_path = OUTPUT_DIR / f"{dept}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"  ✓ {dept}.json: {data['faculty_count']} faculty")
        else:
            print(f"  ✗ {dept}.json: NOT FOUND")
    
    # Check index
    index_file = DATA_DIR / "faculty" / "department_index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index = json.load(f)
        print(f"  ✓ department_index.json: {index['total_departments']} departments")
    else:
        print(f"  ✗ department_index.json: NOT FOUND")


if __name__ == "__main__":
    print("="*50)
    print("Faculty Data Reorganization Script")
    print("="*50)
    print()
    
    success = reorganize_faculty()
    
    if success:
        verify_output()