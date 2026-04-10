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
        """Load all faculty data from combined file (one-time).

        Copies each faculty dict before annotating it with department metadata
        so the original parsed JSON in self._data is not mutated — keeps the
        source data reusable and avoids surprising aliasing bugs.
        """
        if self._loaded:
            return

        with open(self.combined_file, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

        # Build flat list of all faculty with department info (shallow copies)
        for dept_key, dept_data in self._data.items():
            faculty_list = dept_data.get("faculty", [])
            for fac in faculty_list:
                fac_copy = dict(fac)
                fac_copy["department"] = dept_data.get("name", dept_key)
                fac_copy["department_key"] = dept_key
                self._all_faculty.append(fac_copy)

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
            List of faculty members with full details (copies — safe to mutate)
        """
        self._load_data()

        dept_data = self._data.get(dept_key)
        if not dept_data:
            return []

        dept_name = dept_data.get("name", dept_key)
        faculty = [
            {**f, "department": f.get("department", dept_name),
             "department_key": f.get("department_key", dept_key)}
            for f in dept_data.get("faculty", [])
        ]

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
    
    # Map research topics to their "home" departments for ranking.
    # When a topic matches, professors from these departments appear first.
    # Professors from other departments still appear — just ranked below.
    TOPIC_DEPARTMENT_AFFINITY = {
        # CS / Engineering topics → EECS first
        "machine learning": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "ml": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "artificial intelligence": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "ai": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "deep learning": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "dl": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "natural language processing": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "nlp": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "computer vision": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "cv": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "robotics": {"eecs", "electrical_engineering_computer_science", "mechanical_engineering"},
        "cybersecurity": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "data science": {"eecs", "electrical_engineering_computer_science", "computer_science", "math"},
        "software engineering": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "algorithms": {"eecs", "electrical_engineering_computer_science", "computer_science", "math"},
        "programming": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "hci": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "human computer interaction": {"eecs", "electrical_engineering_computer_science", "computer_science"},
        "networking": {"eecs", "electrical_engineering_computer_science"},
        "database": {"eecs", "electrical_engineering_computer_science", "computer_science"},

        # Physics / Astronomy topics → Physics first
        "quantum": {"physics", "physics_astronomy"},
        "quantum computing": {"physics", "physics_astronomy", "eecs"},
        "particle physics": {"physics", "physics_astronomy"},
        "astrophysics": {"physics", "physics_astronomy"},
        "cosmology": {"physics", "physics_astronomy"},
        "condensed matter": {"physics", "physics_astronomy"},
        "optics": {"physics", "physics_astronomy"},
        "thermodynamics": {"physics", "physics_astronomy", "mechanical_engineering"},

        # Chemistry topics → Chemistry first
        "organic chemistry": {"chemistry"},
        "inorganic chemistry": {"chemistry"},
        "biochemistry": {"chemistry", "molecular_biosciences"},
        "computational chemistry": {"chemistry"},
        "materials science": {"chemistry", "chemical_petroleum_engineering"},

        # Biology topics → Biology first
        "genetics": {"molecular_biosciences", "ecology_evolutionary_biology", "biology"},
        "ecology": {"ecology_evolutionary_biology", "biology"},
        "evolution": {"ecology_evolutionary_biology", "biology"},
        "molecular biology": {"molecular_biosciences", "biology"},
        "neuroscience": {"molecular_biosciences", "psychology", "biology"},
        "microbiology": {"molecular_biosciences", "biology"},
        "bioinformatics": {"molecular_biosciences", "eecs", "biology"},

        # Math / Stats topics → Math first
        "statistics": {"math", "mathematics"},
        "probability": {"math", "mathematics"},
        "calculus": {"math", "mathematics"},
        "algebra": {"math", "mathematics"},
        "topology": {"math", "mathematics"},
        "number theory": {"math", "mathematics"},

        # Engineering topics → Respective departments
        "aerospace": {"aerospace_engineering"},
        "aerodynamics": {"aerospace_engineering"},
        "structural engineering": {"civil_environmental_architectural_engineering"},
        "environmental engineering": {"civil_environmental_architectural_engineering"},
        "biomedical engineering": {"bioengineering"},
        "chemical engineering": {"chemical_petroleum_engineering"},
        "petroleum engineering": {"chemical_petroleum_engineering"},
        "mechanical": {"mechanical_engineering"},
        "fluid dynamics": {"mechanical_engineering", "aerospace_engineering"},

        # Business topics → Business first
        "finance": {"business", "school_of_business"},
        "marketing": {"business", "school_of_business"},
        "accounting": {"business", "school_of_business"},
        "management": {"business", "school_of_business"},
        "supply chain": {"business", "school_of_business"},
        "entrepreneurship": {"business", "school_of_business"},
        "analytics": {"business", "school_of_business", "eecs"},

        # Psychology topics → Psychology first
        "clinical psychology": {"psychology"},
        "cognitive": {"psychology"},
        "behavioral": {"psychology"},
        "developmental psychology": {"psychology"},
        "social psychology": {"psychology"},

        # Law topics → Law first
        "constitutional law": {"law"},
        "criminal law": {"law"},
        "international law": {"law"},
        "intellectual property": {"law"},

        # Humanities / Social Sciences
        "political science": {"political_science"},
        "economics": {"economics"},
        "sociology": {"sociology"},
        "anthropology": {"anthropology"},
        "philosophy": {"philosophy"},
        "linguistics": {"linguistics"},
        "history": {"history"},
        "literature": {"english"},
        "journalism": {"journalism"},
        "education": {"education_human_sciences"},
        "music": {"music"},
        "architecture": {"architecture"},
        "pharmacy": {"pharmacy"},
        "nursing": {"nursing"},
        "social work": {"social_welfare"},

        # Health / Medical topics
        "public health": {"public_health", "nursing"},
        "epidemiology": {"public_health"},
        "pharmacology": {"pharmacy", "pharmacology_toxicology"},
        "physical therapy": {"physical_therapy"},
        "occupational therapy": {"occupational_therapy"},
    }

    def _filter_by_research(self, faculty: List[Dict], research_area: str) -> List[Dict]:
        """Filter faculty by research area, ranked by department affinity."""
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

        # Determine which departments should be boosted for this topic
        preferred_depts = set()
        for term in search_terms:
            if term in self.TOPIC_DEPARTMENT_AFFINITY:
                preferred_depts.update(self.TOPIC_DEPARTMENT_AFFINITY[term])

        primary_matches = []    # From preferred departments
        secondary_matches = []  # From other departments

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
            matched = False
            for term in search_terms:
                if term in searchable:
                    matched = True
                    break

            if matched:
                dept_key = f.get("department_key", "").lower()
                if preferred_depts and dept_key in preferred_depts:
                    primary_matches.append(f)
                else:
                    secondary_matches.append(f)

        # Return preferred department matches first, then others
        return primary_matches + secondary_matches
    
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
    
    def search_by_research_keywords(self, keywords: List[str], department_key: str = None,
                                     limit: int = 10) -> List[Dict]:
        """
        Find faculty whose research_interests contain ALL of the given keywords
        (case-insensitive substring match). Useful to supplement vector search.

        Args:
            keywords: List of keyword strings to match (e.g. ["machine", "learning"])
            department_key: Optional dept abbreviation filter (e.g. "eecs")
            limit: Max results

        Returns:
            List of faculty dicts that match
        """
        self._load_data()
        results = []
        for f in self._all_faculty:
            if department_key:
                if f.get("department_key", "") != department_key.lower():
                    continue
            research = " ".join(str(r) for r in (f.get("research_interests") or [])).lower()
            if all(kw.lower() in research for kw in keywords):
                results.append(f)
        return results[:limit]

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