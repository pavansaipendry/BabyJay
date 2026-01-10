"""
Faculty Retriever - Fast Direct JSON Lookup
============================================
Loads all faculty from single combined file for fast access.

Speed comparison:
- Vector search: ~700ms (OpenAI API call for embeddings)
- Direct lookup: ~5-20ms (single file load + filter)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class FacultyRetriever:
    """Fast faculty search using single combined JSON file."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the faculty retriever.
        
        Args:
            data_dir: Path to data directory (defaults to BabyJay/data)
        """
        if data_dir is None:
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "data").exists():
                    data_dir = str(current / "data")
                    break
                current = current.parent
            else:
                raise FileNotFoundError("Could not find data directory")
        
        self.data_dir = Path(data_dir)
        self.combined_file = self.data_dir / "all_faculty_combined.json"
        
        # Data storage
        self._data: Dict[str, Dict] = {}
        self._all_faculty: List[Dict] = []
        self._loaded = False
        
        # Verify data exists
        if not self.combined_file.exists():
            raise FileNotFoundError(f"Combined faculty file not found: {self.combined_file}")
    
    def _load_data(self):
        """Load all faculty data from combined file (one-time)."""
        if self._loaded:
            return
        
        with open(self.combined_file, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
        
        # Build flat list of all faculty with department info
        for dept_key, dept_data in self._data.items():
            faculty_list = dept_data.get("faculty", [])
            for fac in faculty_list:
                fac["department"] = dept_data.get("name", dept_key)
                fac["department_key"] = dept_key
                self._all_faculty.append(fac)
        
        self._loaded = True
    
    def get_all_departments(self) -> List[Dict]:
        """Get list of all departments with metadata."""
        self._load_data()
        
        departments = []
        for dept_key, dept_data in self._data.items():
            departments.append({
                "key": dept_key,
                "name": dept_data.get("name", dept_key),
                "faculty_count": dept_data.get("faculty_count", len(dept_data.get("faculty", [])))
            })
        return departments
    
    def get_department_faculty(self, dept_key: str, limit: int = None) -> List[Dict]:
        """
        Get all faculty in a department.
        
        Args:
            dept_key: Department key (e.g., "eecs", "physics")
            limit: Optional limit on results
            
        Returns:
            List of faculty members with full details
        """
        self._load_data()
        
        dept_data = self._data.get(dept_key)
        if not dept_data:
            return []
        
        faculty = dept_data.get("faculty", [])
        
        # Ensure department info is set
        for f in faculty:
            if "department" not in f:
                f["department"] = dept_data.get("name", dept_key)
            if "department_key" not in f:
                f["department_key"] = dept_key
        
        if limit:
            return faculty[:limit]
        return faculty
    
    def search(self, 
               department: str = None, 
               research_area: str = None,
               limit: int = None,
               scope: str = "top_results") -> List[Dict]:
        """
        Search for faculty members.
        
        Args:
            department: Department key to search within (optional)
            research_area: Research area/topic to filter by (optional)
            limit: Maximum results to return
            scope: "top_results" (default 10) or "complete_list" (all matches)
            
        Returns:
            List of matching faculty members
        """
        self._load_data()
        
        # Set default limit based on scope
        if limit is None:
            limit = 100 if scope == "complete_list" else 10
        
        results = []
        
        if department:
            # Search within specific department
            faculty = self.get_department_faculty(department)
            if research_area:
                faculty = self._filter_by_research(faculty, research_area)
            results = faculty
        elif research_area:
            # Search across all departments
            results = self._filter_by_research(self._all_faculty, research_area)
        else:
            # No filters - return empty (too broad)
            return []
        
        # Deduplicate by name (same person might be in multiple departments)
        seen_names = set()
        deduped = []
        for f in results:
            name = f.get("name", "").lower()
            if name not in seen_names:
                seen_names.add(name)
                deduped.append(f)
        
        return deduped[:limit]
    
    def _filter_by_research(self, faculty: List[Dict], research_area: str) -> List[Dict]:
        """Filter faculty by research area."""
        research_lower = research_area.lower()
        
        # Expand common abbreviations
        expansions = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "dl": "deep learning",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "hci": "human computer interaction",
        }
        
        # Build search terms
        search_terms = [research_lower]
        for abbr, full in expansions.items():
            if research_lower == abbr:
                search_terms.append(full)
            elif research_lower == full:
                search_terms.append(abbr)
        
        matches = []
        for f in faculty:
            # Get searchable text
            research_interests = f.get("research_interests", [])
            if isinstance(research_interests, list):
                research_text = " ".join(research_interests).lower()
            else:
                research_text = str(research_interests).lower()
            
            bio = f.get("biography", "").lower()
            title = f.get("title", "").lower()
            
            searchable = f"{research_text} {bio} {title}"
            
            # Check if any search term matches
            for term in search_terms:
                if term in searchable:
                    matches.append(f)
                    break
        
        return matches
    
    def search_by_name(self, name: str, limit: int = 5) -> List[Dict]:
        """
        Find faculty by name.
        
        Args:
            name: Full or partial name to search
            limit: Maximum results
            
        Returns:
            List of matching faculty
        """
        self._load_data()
        
        name_lower = name.lower()
        matches = []
        
        for f in self._all_faculty:
            faculty_name = f.get("name", "").lower()
            if name_lower in faculty_name:
                matches.append(f)
        
        # Deduplicate
        seen_names = set()
        deduped = []
        for f in matches:
            fname = f.get("name", "").lower()
            if fname not in seen_names:
                seen_names.add(fname)
                deduped.append(f)
        
        return deduped[:limit]
    
    def format_for_context(self, faculty_list: List[Dict]) -> str:
        """Format faculty list as context string for LLM."""
        if not faculty_list:
            return ""
        
        lines = ["=== FACULTY INFORMATION ==="]
        
        for f in faculty_list:
            research = f.get("research_interests", [])
            if isinstance(research, list):
                research_str = " ".join(research)
            else:
                research_str = str(research)
            
            lines.append(f"""
Professor: {f.get('name', 'Unknown')}
Department: {f.get('department', f.get('department_key', 'Unknown'))}
Email: {f.get('email', 'N/A')}
Phone: {f.get('phone', 'N/A')}
Office: {f.get('office', 'N/A')}
Profile: {f.get('profile_url', 'N/A')}
Research: {research_str[:500]}""")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict:
        """Get statistics about the faculty data."""
        self._load_data()
        
        return {
            "total_departments": len(self._data),
            "total_faculty": len(self._all_faculty),
            "data_file": str(self.combined_file)
        }


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing Faculty Retriever")
    print("=" * 60)
    
    start = time.time()
    retriever = FacultyRetriever()
    stats = retriever.get_stats()
    load_time = (time.time() - start) * 1000
    
    print(f"\nData loaded in {load_time:.0f}ms")
    print(f"  Departments: {stats['total_departments']}")
    print(f"  Total Faculty: {stats['total_faculty']}")
    
    # Test 1: Get all EECS faculty
    print("\n" + "=" * 60)
    print("Test 1: All EECS faculty")
    start = time.time()
    results = retriever.get_department_faculty("eecs")
    elapsed = (time.time() - start) * 1000
    print(f"  Found: {len(results)} faculty in {elapsed:.1f}ms")
    for f in results[:3]:
        print(f"    - {f['name']}")
    
    # Test 2: EECS + ML filter
    print("\n" + "=" * 60)
    print("Test 2: EECS faculty doing ML research")
    start = time.time()
    results = retriever.search(department="eecs", research_area="machine learning")
    elapsed = (time.time() - start) * 1000
    print(f"  Found: {len(results)} faculty in {elapsed:.1f}ms")
    for f in results[:5]:
        print(f"    - {f['name']}")
    
    # Test 3: ML across all departments
    print("\n" + "=" * 60)
    print("Test 3: ML researchers (all departments)")
    start = time.time()
    results = retriever.search(research_area="machine learning", scope="complete_list")
    elapsed = (time.time() - start) * 1000
    print(f"  Found: {len(results)} faculty in {elapsed:.1f}ms")
    for f in results[:5]:
        print(f"    - {f['name']} ({f.get('department', 'Unknown')})")
    
    # Test 4: Search by name
    print("\n" + "=" * 60)
    print("Test 4: Search by name 'Li'")
    start = time.time()
    results = retriever.search_by_name("Li")
    elapsed = (time.time() - start) * 1000
    print(f"  Found: {len(results)} faculty in {elapsed:.1f}ms")
    for f in results[:5]:
        print(f"    - {f['name']} ({f.get('department', 'Unknown')})")
    
    print("\n" + "=" * 60)
    print("All tests complete!")