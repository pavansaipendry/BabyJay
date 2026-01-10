"""
Convert all_faculty_combined.json to faculty_documents.json
============================================================
Run this after updating all_faculty_combined.json

USAGE (from BabyJay root):
    python app/scripts/convert_faculty_data.py
"""

import json
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def convert_faculty_data():
    """Convert combined faculty JSON to searchable documents format."""
    
    input_path = os.path.join(DATA_DIR, 'all_faculty_combined.json')
    output_path = os.path.join(DATA_DIR, 'faculty_documents.json')
    
    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create flattened faculty documents
    faculty_documents = []
    removed_count = 0
    
    for dept_key, dept_data in data.items():
        dept_name = dept_data.get('name', '')
        building = dept_data.get('building', '')
        dept_address = dept_data.get('address', '')
        dept_phone = dept_data.get('phone', '')
        dept_email = dept_data.get('email', '')
        dept_website = dept_data.get('website', '')
        
        for faculty in dept_data.get('faculty', []):
            name = faculty.get('name', '').strip()
            
            # Skip invalid entries (SSO, empty names, etc.)
            if not name:
                removed_count += 1
                continue
            if 'Sign-On' in name or 'SSO' in name:
                removed_count += 1
                continue
            
            # Create document ID
            doc_id = f"faculty_{dept_key}_{name.lower().replace(' ', '_').replace(',', '').replace('.', '')}"
            
            # Build searchable text
            research = ' '.join(faculty.get('research_interests', []))
            bio = faculty.get('biography', '')
            
            searchable_text = f"""
Professor {name}
Department: {dept_name}
Title: {faculty.get('title', '')}
Office: {faculty.get('office', '')} in {building}
Phone: {faculty.get('phone', '')}
Email: {faculty.get('email', '')}
Research Interests: {research}
Biography: {bio}
""".strip()
            
            doc = {
                "id": doc_id,
                "type": "faculty",
                "department": dept_name,
                "department_key": dept_key,
                "department_building": building,
                "department_contact": {
                    "address": dept_address,
                    "phone": dept_phone,
                    "email": dept_email,
                    "website": dept_website
                },
                "name": name,
                "title": faculty.get('title', ''),
                "office": faculty.get('office', ''),
                "phone": faculty.get('phone', ''),
                "email": faculty.get('email', ''),
                "website": faculty.get('website', ''),
                "profile_url": faculty.get('profile_url', ''),
                "research_interests": faculty.get('research_interests', []),
                "biography": bio,
                "searchable_text": searchable_text
            }
            faculty_documents.append(doc)
    
    # Save
    print(f"Saving: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(faculty_documents, f, indent=2)
    
    print(f"\n✓ Created {len(faculty_documents)} faculty documents")
    if removed_count > 0:
        print(f"  (Removed {removed_count} invalid entries)")
    
    return len(faculty_documents)

if __name__ == '__main__':
    print("=" * 60)
    print("Faculty Data Converter")
    print("=" * 60)
    convert_faculty_data()
    print("\nNext step: Run generate_faculty_embeddings.py")