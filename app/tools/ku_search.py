"""
KU Search Integration
Uses Google Custom Search Engine to search KU websites
Similar to live_course_lookup.py but for general KU content
"""

import json
import re
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class KUSearchAPI:
    """
    Search KU using Google CSE with proper browser headers/cookies
    """
    
    def __init__(self, cookies: Optional[Dict[str, str]] = None):
        self.base_url = "https://cse.google.com/cse/element/v1"
        self.cse_id = "005110938966103991125:gjdigwfiiea"
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes
        
        # Store cookies if provided
        self.cookies = cookies or {}
        
        # Session for persistent cookies
        self.session = requests.Session()
        if self.cookies:
            self.session.cookies.update(self.cookies)
    
    def search(self, query: str, num_results: int = 5, site: str = "ku.edu") -> List[Dict]:
        """
        Search KU websites for a query
        
        Args:
            query: Search query (e.g. "OPT application")
            num_results: Number of results to return (1-10)
            site: Site to restrict search to (default: ku.edu)
        
        Returns:
            List of dicts with keys: title, url, snippet
        """
        # Check cache first
        cache_key = f"{query}_{num_results}_{site}".lower()
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                print(f"[KU Search] Cache hit for: {query}")
                return cached_result
        
        print(f"[KU Search] Fetching fresh results for: {query}")
        
        # Build request parameters (without cse_tok first)
        params = {
            "rsz": "filtered_cse",
            "num": str(num_results),
            "hl": "en",
            "source": "gcsc",
            "cselibv": "f71e4ed980f4c082",
            "cx": self.cse_id,
            "q": query,
            "safe": "off",
            "as_sitesearch": site,
            "sort": "",
            "exp": "cc,apo",
            "callback": "google.search.cse.api0000",  # Use a consistent callback
            "rurl": f"https://ku.edu/search?q={query}",
        }
        
        # Full browser-like headers
        headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": "https://ku.edu/",
            "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "script",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        }
        
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            # Parse JSONP response
            results = self._parse_jsonp_response(response.text)
            
            # Cache the results
            self.cache[cache_key] = (results, datetime.now())
            
            return results
            
        except Exception as e:
            print(f"[KU Search] Error: {e}")
            return []
    
    def set_cookies_from_browser(self, cookie_dict: Dict[str, str]):
        """
        Set cookies from browser (after copying from DevTools)
        
        Args:
            cookie_dict: Dictionary of cookie names to values
        """
        self.cookies = cookie_dict
        self.session.cookies.update(cookie_dict)
        print(f"[KU Search] Updated cookies: {len(cookie_dict)} cookies set")
    
    def _parse_jsonp_response(self, text: str) -> List[Dict]:
        """
        Parse JSONP response and extract search results
        
        Format: google.search.cse.apiXXXX({...});
        """
        # Try multiple JSONP patterns
        patterns = [
            r"google\.search\.cse\.\w+\((.*)\)\s*;?\s*$",  # google.search.cse.api0000(...)
            r"__cse_callback\((.*)\)\s*;?\s*$",  # __cse_callback(...)
        ]
        
        match = None
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                break
        
        if not match:
            # Print first 200 chars for debugging
            print(f"[KU Search] Could not parse JSONP. First 200 chars: {text[:200]}")
            raise ValueError("Could not parse JSONP response")
        
        # Parse JSON
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            print(f"[KU Search] JSON decode error: {e}")
            print(f"[KU Search] Extracted JSON (first 500 chars): {match.group(1)[:500]}")
            raise
        
        # Extract results
        results = []
        for item in data.get("results", []):
            result = {
                "title": item.get("titleNoFormatting", ""),
                "url": item.get("unescapedUrl", ""),
                "snippet": item.get("contentNoFormatting", ""),
                "visible_url": item.get("visibleUrl", ""),
            }
            results.append(result)
        
        return results
    
    def search_with_context(self, query: str, num_results: int = 3) -> Dict:
        """
        Search and return formatted context for LLM
        
        Returns dict with:
            - query: original query
            - results: list of search results
            - sources: list of URLs
            - context_text: formatted text for LLM
        """
        results = self.search(query, num_results)
        
        # Format context for LLM
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result['title']}")
            context_parts.append(f"   URL: {result['url']}")
            context_parts.append(f"   {result['snippet']}")
            context_parts.append("")
            sources.append(result['url'])
        
        return {
            "query": query,
            "results": results,
            "sources": sources,
            "context_text": "\n".join(context_parts),
            "num_found": len(results)
        }


# Example usage
if __name__ == "__main__":
    # Method 1: Try without cookies first (might work, might get 403)
    print("=" * 60)
    print("Method 1: Testing WITHOUT cookies")
    print("=" * 60)
    ku_search = KUSearchAPI()
    results = ku_search.search("OPT", num_results=3)
    
    if results:
        print("\n✅ SUCCESS! No cookies needed.\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   {result['url']}")
            print(f"   {result['snippet'][:100]}...\n")
    else:
        print("\n❌ Failed without cookies. Need to add browser cookies.\n")
        print("=" * 60)
        print("Method 2: Using browser cookies")
        print("=" * 60)
        print("\nTo get cookies:")
        print("1. Open Chrome DevTools on ku.edu/search")
        print("2. Network tab → Find the cse/element/v1 request")
        print("3. Right-click → Copy → Copy as cURL")
        print("4. Extract cookies from the -H 'cookie: ...' line")
        print("\nThen use them like this:\n")
        print("cookies = {")
        print('    "__Secure-3PAPISID": "YOUR_VALUE",')
        print('    "__Secure-3PSID": "YOUR_VALUE",')
        print('    "NID": "YOUR_VALUE",')
        print('    "__Secure-3PSIDCC": "YOUR_VALUE",')
        print("}")
        print("ku_search = KUSearchAPI(cookies=cookies)")
        print("results = ku_search.search('OPT')")
    
    # Test with context formatting (if results exist)
    if results:
        print("\n" + "=" * 60)
        print("Test: Search with formatted context for LLM")
        print("=" * 60)
        context = ku_search.search_with_context("how to apply to OPT", num_results=2)
        print(f"Found {context['num_found']} results")
        print("\nFormatted context:")
        print(context['context_text'])