"""
Campus Data Retriever for BabyJay
=================================
Fast direct JSON lookup for dining, transit, housing, tuition, and other campus data.

Speed: ~1-5ms per query (vs 500-1000ms with ChromaDB)

Usage:
    from app.rag.campus_retriever import CampusRetriever
    
    retriever = CampusRetriever()
    results = retriever.search_dining("coffee")
    results = retriever.search_transit("campus")
    results = retriever.search_housing("scholarship")
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class CampusRetriever:
    """Fast JSON-based retriever for campus data."""
    
    def __init__(self, data_dir: str = None):
        """Initialize with data directory."""
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
        self._cache: Dict[str, Any] = {}
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file with caching."""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._cache[filename] = data
        return data
    
    def _partial_match(self, text: str, query: str) -> bool:
        """Check if any query term appears in text."""
        if not text or not query:
            return False
        
        text_lower = text.lower()
        query_terms = query.lower().split()
        return any(term in text_lower for term in query_terms)
    
    # ==================== DINING ====================
    
    def search_dining(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search dining locations."""
        data = self._load_json("dining/locations.json")
        locations = data.get("locations", [])
        
        if not query:
            return locations[:limit]
        
        results = []
        for loc in locations:
            searchable = f"{loc.get('name', '')} {loc.get('building', '')} {loc.get('type', '')} {loc.get('description', '')}"
            if self._partial_match(searchable, query):
                results.append(loc)
        
        return results[:limit]
    
    def get_all_dining(self) -> List[Dict]:
        """Get all dining locations."""
        data = self._load_json("dining/locations.json")
        return data.get("locations", [])
    
    def format_dining_context(self, locations: List[Dict]) -> str:
        """Format dining results for LLM context."""
        if not locations:
            return ""
        
        lines = ["=== DINING INFORMATION ==="]
        for loc in locations:
            hours = loc.get('hours', {})
            hours_str = f"Mon-Fri: {hours.get('monday_friday', 'N/A')}, Sat-Sun: {hours.get('saturday_sunday', 'N/A')}"
            
            lines.append(f"""
Name: {loc.get('name', 'Unknown')}
Building: {loc.get('building', 'N/A')}
Address: {loc.get('address', 'N/A')}
Type: {loc.get('type', 'N/A')}
Description: {loc.get('description', 'N/A')}
Hours: {hours_str}""")
        
        return "\n".join(lines)
    
    # ==================== TRANSIT ====================
    
    def search_transit(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search transit routes."""
        data = self._load_json("transit/routes.json")
        routes = data.get("routes", [])
        
        if not query:
            return routes[:limit]
        
        results = []
        query_lower = query.lower()
        
        for route in routes:
            # Build searchable text from actual fields
            searchable = f"{route.get('route_name', '')} {route.get('route_number', '')} {route.get('description', '')} {route.get('operates', '')}"
            
            # Check for KU-specific queries
            if 'ku' in query_lower or 'campus' in query_lower:
                if route.get('serves_ku', False) or route.get('campus_only', False):
                    results.append(route)
                    continue
            
            if self._partial_match(searchable, query):
                results.append(route)
        
        return results[:limit]
    
    def get_all_transit(self) -> List[Dict]:
        """Get all transit routes."""
        data = self._load_json("transit/routes.json")
        return data.get("routes", [])
    
    def get_ku_transit(self) -> List[Dict]:
        """Get only KU campus routes."""
        data = self._load_json("transit/routes.json")
        routes = data.get("routes", [])
        return [r for r in routes if r.get('serves_ku', False) or r.get('campus_only', False)]
    
    def format_transit_context(self, routes: List[Dict]) -> str:
        """Format transit results for LLM context."""
        if not routes:
            return ""
        
        lines = ["=== TRANSIT INFORMATION ==="]
        for route in routes:
            serves_ku = "Yes" if route.get('serves_ku', False) else "No"
            campus_only = "Yes" if route.get('campus_only', False) else "No"
            
            lines.append(f"""
Route: {route.get('route_number', 'N/A')} - {route.get('route_name', 'Unknown')}
Description: {route.get('description', 'N/A')}
Operates: {route.get('operates', 'N/A')}
Serves KU: {serves_ku}
Campus Only: {campus_only}""")
        
        return "\n".join(lines)
    
    # ==================== HOUSING ====================
    
    def search_housing(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search housing options."""
        data = self._load_json("housing/housing.json")
        housing_data = data.get("housing", {})
        
        # Collect all housing locations from nested structure
        all_housing = []
        
        # General info
        general = housing_data.get("general_info", {})
        if general:
            all_housing.append({
                "type": "general_info",
                "name": "General Housing Information",
                **general
            })
        
        # Residence halls
        res_halls = housing_data.get("residence_halls", {})
        for loc in res_halls.get("locations", []):
            loc["category"] = "Residence Hall"
            all_housing.append(loc)
        
        # Scholarship halls
        scholarship = housing_data.get("scholarship_halls", {})
        for loc in scholarship.get("locations", scholarship.get("halls", [])):
            loc["category"] = "Scholarship Hall"
            all_housing.append(loc)
        
        # Apartments
        apartments = housing_data.get("apartments", {})
        for loc in apartments.get("locations", apartments.get("complexes", [])):
            loc["category"] = "Apartment"
            all_housing.append(loc)
        
        if not query:
            return all_housing[:limit]
        
        results = []
        for h in all_housing:
            searchable = f"{h.get('name', '')} {h.get('type', '')} {h.get('category', '')} {h.get('description', '')}"
            if self._partial_match(searchable, query):
                results.append(h)
        
        return results[:limit]
    
    def get_all_housing(self) -> List[Dict]:
        """Get all housing options."""
        return self.search_housing(None, limit=100)
    
    def format_housing_context(self, housing: List[Dict]) -> str:
        """Format housing results for LLM context."""
        if not housing:
            return ""
        
        lines = ["=== HOUSING INFORMATION ==="]
        for h in housing:
            if h.get('type') == 'general_info':
                lines.append(f"""
General Information:
Description: {h.get('description', 'N/A')}
Application Fee: {h.get('application_fee', 'N/A')}
Dining Plan Required: {h.get('dining_plan_required', 'N/A')}""")
            else:
                # Format rates if available
                rates = h.get('rates_2026_27', h.get('rates', {}))
                if isinstance(rates, dict):
                    rates_str = ", ".join([f"{k}: {v}" for k, v in list(rates.items())[:3]])
                else:
                    rates_str = str(rates) if rates else "N/A"
                
                lines.append(f"""
Name: {h.get('name', 'Unknown')}
Category: {h.get('category', 'N/A')}
Type: {h.get('type', 'N/A')}
Room Types: {h.get('room_types', 'N/A')}
Bath: {h.get('bath', 'N/A')}
Rates: {rates_str}""")
        
        return "\n".join(lines)
    
    # ==================== TUITION ====================
    
    def search_tuition(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search tuition and fee information."""
        data = self._load_json("tuition/tuition_fees.json")
        
        # Flatten the structure
        fees = []
        
        def extract_fees(obj, category=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['last_updated', 'source', 'academic_year']:
                        continue
                    if isinstance(value, (dict, list)):
                        extract_fees(value, key)
                    else:
                        fees.append({"category": category, "item": key, "value": value})
            elif isinstance(obj, list):
                for item in obj:
                    extract_fees(item, category)
        
        extract_fees(data)
        
        if not query:
            return fees[:limit]
        
        results = []
        for fee in fees:
            searchable = f"{fee.get('category', '')} {fee.get('item', '')} {fee.get('value', '')}"
            if self._partial_match(searchable, query):
                results.append(fee)
        
        return results[:limit]
    
    def get_all_tuition(self) -> List[Dict]:
        """Get all tuition/fee information."""
        return self.search_tuition(None, limit=100)
    
    def format_tuition_context(self, fees: List[Dict]) -> str:
        """Format tuition results for LLM context."""
        if not fees:
            return ""
        
        lines = ["=== TUITION & FEES INFORMATION ==="]
        current_category = None
        
        for fee in fees:
            cat = fee.get('category', 'General')
            if cat != current_category:
                lines.append(f"\n[{cat}]")
                current_category = cat
            lines.append(f"  {fee.get('item', 'N/A')}: {fee.get('value', 'N/A')}")
        
        return "\n".join(lines)
    
    # ==================== GENERIC SEARCH ====================
    
    def search(self, data_type: str, query: str = None, limit: int = 10) -> Dict[str, Any]:
        """
        Generic search across any data type.
        
        Args:
            data_type: One of 'dining', 'transit', 'housing', 'tuition'
            query: Search term
            limit: Max results
            
        Returns:
            Dict with 'results' and 'context'
        """
        search_methods = {
            'dining': (self.search_dining, self.format_dining_context),
            'transit': (self.search_transit, self.format_transit_context),
            'housing': (self.search_housing, self.format_housing_context),
            'tuition': (self.search_tuition, self.format_tuition_context),
        }
        
        if data_type not in search_methods:
            return {"results": [], "context": "", "result_count": 0}
        
        search_fn, format_fn = search_methods[data_type]
        results = search_fn(query, limit)
        context = format_fn(results)
        
        return {
            "results": results,
            "context": context,
            "result_count": len(results)
        }


# Quick test
if __name__ == "__main__":
    import time
    
    print("Testing Campus Retriever")
    print("=" * 60)
    
    retriever = CampusRetriever()
    
    # Test Dining
    print("\n=== DINING ===")
    start = time.time()
    results = retriever.search_dining("dining")
    elapsed = (time.time() - start) * 1000
    print(f"Query: 'dining' - {len(results)} results in {elapsed:.1f}ms")
    for r in results[:3]:
        print(f"  - {r.get('name')} ({r.get('building')})")
    
    all_dining = retriever.get_all_dining()
    print(f"All dining: {len(all_dining)} locations")
    
    # Test Transit
    print("\n=== TRANSIT ===")
    start = time.time()
    results = retriever.search_transit("KU campus")
    elapsed = (time.time() - start) * 1000
    print(f"Query: 'KU campus' - {len(results)} results in {elapsed:.1f}ms")
    for r in results[:3]:
        print(f"  - Route {r.get('route_number')}: {r.get('route_name')}")
    
    ku_routes = retriever.get_ku_transit()
    print(f"KU routes: {len(ku_routes)} routes serve campus")
    
    # Test Housing
    print("\n=== HOUSING ===")
    start = time.time()
    results = retriever.search_housing("scholarship")
    elapsed = (time.time() - start) * 1000
    print(f"Query: 'scholarship' - {len(results)} results in {elapsed:.1f}ms")
    for r in results[:3]:
        print(f"  - {r.get('name')} ({r.get('category', 'N/A')})")
    
    start = time.time()
    results = retriever.search_housing("residence")
    elapsed = (time.time() - start) * 1000
    print(f"Query: 'residence' - {len(results)} results in {elapsed:.1f}ms")
    
    # Test Tuition
    print("\n=== TUITION ===")
    start = time.time()
    results = retriever.search_tuition("undergraduate")
    elapsed = (time.time() - start) * 1000
    print(f"Query: 'undergraduate' - {len(results)} results in {elapsed:.1f}ms")
    for r in results[:3]:
        print(f"  - [{r.get('category')}] {r.get('item')}: {r.get('value')}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")