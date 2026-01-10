import json
from datetime import datetime

# Lawrence Transit data collected from lawrencetransit.org
transit_data = {
    "last_updated": datetime.now().isoformat(),
    "routes": [
        {
            "route_number": "1",
            "route_name": "East 7th Street",
            "description": "Serves East Lawrence via 7th Street",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "Vermont @ 7th",
                "Mass @ 19th", 
                "23rd @ Harper",
                "Lawrence Community Shelter",
                "Greenway Circle"
            ],
            "serves_ku": False
        },
        {
            "route_number": "2",
            "route_name": "North Lawrence",
            "description": "Serves North Lawrence and LMH hospital",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "Central Station",
                "Rockledge @ 6th",
                "LMH",
                "Lakeview"
            ],
            "serves_ku": False
        },
        {
            "route_number": "3",
            "route_name": "North Rockledge",
            "description": "Serves North Lawrence via Rockledge",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "Vermont @ 7th",
                "Rockledge @ 6th",
                "LMH",
                "Lakeview"
            ],
            "serves_ku": True
        },
        {
            "route_number": "4",
            "route_name": "Wakarusa/6th via Central & KU",
            "description": "Major route serving West Lawrence and KU campus",
            "operates": "Monday-Friday",
            "frequency": "30 minutes",
            "key_stops": [
                "Wakarusa @ 6th",
                "Wakarusa @ Bob Billings",
                "Central Station",
                "Chi Omega Fountain (KU)",
                "Vermont @ 7th"
            ],
            "serves_ku": True,
            "popular_for_students": True
        },
        {
            "route_number": "5",
            "route_name": "East 23rd Street",
            "description": "Serves East Lawrence via 23rd Street",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "Central Station",
                "24th @ Crestline",
                "23rd @ Mass",
                "23rd @ Harper",
                "Greenway Circle"
            ],
            "serves_ku": False
        },
        {
            "route_number": "6",
            "route_name": "West 6th Street",
            "description": "Serves West Lawrence and Free State High School",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "LMH West",
                "Free State High",
                "6th @ Lawrence",
                "Vermont @ 7th (KU)",
                "DMV"
            ],
            "serves_ku": True
        },
        {
            "route_number": "7",
            "route_name": "Haskell/Louisiana",
            "description": "Serves Haskell Indian Nations University area",
            "operates": "Monday-Friday",
            "frequency": "30 minutes",
            "key_stops": [
                "31st @ Iowa",
                "Louisiana @ 23rd",
                "Haskell @ 16th",
                "Vermont @ 7th (KU)"
            ],
            "serves_ku": True
        },
        {
            "route_number": "8",
            "route_name": "Peaslee Tech",
            "description": "Serves Peaslee Technical Center and HINU",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "Peaslee Tech",
                "HINU",
                "Naismith @ 15th",
                "Central Station"
            ],
            "serves_ku": False
        },
        {
            "route_number": "9",
            "route_name": "East 27th Street",
            "description": "Serves East Lawrence neighborhoods",
            "operates": "Monday-Friday",
            "frequency": "30-60 minutes",
            "key_stops": [
                "24th @ Crossgate",
                "Lawrence @ 27th",
                "31st @ Iowa"
            ],
            "serves_ku": False
        },
        {
            "route_number": "11",
            "route_name": "Central Station-31st/Iowa via KU",
            "description": "Major route connecting South Lawrence to KU campus",
            "operates": "Monday-Friday, extended hours",
            "frequency": "30 minutes (15 minutes peak)",
            "key_stops": [
                "31st @ Iowa",
                "24th @ Naismith",
                "Wescoe Hall (KU)",
                "Central Station"
            ],
            "serves_ku": True,
            "popular_for_students": True,
            "late_night": "Runs until 10:30 PM"
        },
        {
            "route_number": "12",
            "route_name": "Crossgate",
            "description": "Serves West Lawrence via Bob Billings",
            "operates": "Monday-Friday, extended hours",
            "frequency": "30 minutes",
            "key_stops": [
                "24th @ Crossgate",
                "Bob Billings @ Kasold",
                "Central Station"
            ],
            "serves_ku": False,
            "late_night": "Runs until 10:30 PM"
        },
        {
            "route_number": "30",
            "route_name": "Chelsea Place",
            "description": "KU campus circulator serving student housing",
            "operates": "Monday-Friday during semester",
            "frequency": "20 minutes",
            "key_stops": [
                "Chelsea Place",
                "Kansas Union",
                "Spencer Museum of Art"
            ],
            "serves_ku": True,
            "campus_only": True,
            "popular_for_students": True
        },
        {
            "route_number": "34",
            "route_name": "West 7th Street",
            "description": "KU campus circulator",
            "operates": "Monday-Friday during semester",
            "frequency": "20 minutes",
            "key_stops": [
                "Kansas Union",
                "W 7th Street",
                "Spencer Museum of Art"
            ],
            "serves_ku": True,
            "campus_only": True
        },
        {
            "route_number": "36",
            "route_name": "Frontier & Trail",
            "description": "Serves West Campus student housing",
            "operates": "Monday-Friday during semester",
            "frequency": "20 minutes",
            "key_stops": [
                "Frontier & Trail",
                "9th & Emery",
                "Spencer Museum of Art",
                "Kansas Union"
            ],
            "serves_ku": True,
            "campus_only": True,
            "popular_for_students": True
        },
        {
            "route_number": "38",
            "route_name": "Reserve",
            "description": "Serves West Campus apartments",
            "operates": "Monday-Friday during semester",
            "frequency": "20 minutes",
            "key_stops": [
                "W 31st Street (Reserve)",
                "25th & Melrose",
                "19th & Ousdahl",
                "Kansas Union",
                "Spencer Museum"
            ],
            "serves_ku": True,
            "campus_only": True,
            "popular_for_students": True
        },
        {
            "route_number": "42",
            "route_name": "Daisy Hill Clockwise",
            "description": "KU campus loop via Jayhawk Boulevard (clockwise)",
            "operates": "Monday-Friday during semester",
            "frequency": "5-12 minutes (very frequent)",
            "key_stops": [
                "Daisy Hill (residence halls)",
                "Jayhawk Boulevard",
                "Kansas Union",
                "GSP Hall"
            ],
            "serves_ku": True,
            "campus_only": True,
            "popular_for_students": True,
            "note": "Most frequent route, runs every 5-8 minutes midday"
        },
        {
            "route_number": "43",
            "route_name": "Daisy Hill Counter-Clockwise",
            "description": "KU campus loop via Jayhawk Boulevard (counter-clockwise)",
            "operates": "Monday-Friday during semester",
            "frequency": "9-12 minutes",
            "key_stops": [
                "GSP Hall",
                "Kansas Union",
                "Jayhawk Boulevard",
                "Daisy Hill (residence halls)"
            ],
            "serves_ku": True,
            "campus_only": True,
            "popular_for_students": True
        },
        {
            "route_number": "44",
            "route_name": "Becker Drive Evening",
            "description": "Evening service to Becker Drive apartments",
            "operates": "Monday-Thursday evenings during semester",
            "frequency": "20 minutes",
            "hours": "5:40 PM - 10:00 PM",
            "key_stops": [
                "Becker Drive",
                "18th & Naismith",
                "Kansas Union"
            ],
            "serves_ku": True,
            "campus_only": True,
            "evening_only": True
        },
        {
            "route_number": "45",
            "route_name": "Becker Drive Day",
            "description": "Daytime service to Becker Drive apartments",
            "operates": "Monday-Friday during semester",
            "frequency": "20 minutes",
            "hours": "7:17 AM - 5:30 PM",
            "key_stops": [
                "Becker Drive",
                "18th & Naismith",
                "Kansas Union"
            ],
            "serves_ku": True,
            "campus_only": True
        },
        {
            "route_number": "53",
            "route_name": "SafeRide",
            "description": "Late night KU campus safety shuttle",
            "operates": "Monday-Thursday during semester only",
            "frequency": "20 minutes",
            "hours": "10:00 PM - 1:00 AM",
            "key_stops": [
                "Becker Drive",
                "18th & Naismith",
                "Kansas Union"
            ],
            "serves_ku": True,
            "campus_only": True,
            "late_night": True,
            "popular_for_students": True,
            "note": "Safety shuttle for late night transportation on campus"
        }
    ],
    "important_info": {
        "free_for_ku_students": True,
        "payment_methods": ["Cash", "Mobile app", "KU ID"],
        "real_time_tracking": "Available via Transit app or My Bus Lawrence app",
        "central_station": "7th & Vermont (Downtown Lawrence)",
        "semester_only_routes": ["30", "34", "36", "38", "42", "43", "44", "45", "53"],
        "most_useful_for_students": ["4", "11", "42", "43", "53"]
    }
}

# Save to JSON file
output_path = "data/transit/routes.json"
with open(output_path, 'w') as f:
    json.dump(transit_data, f, indent=2)

print(f"Saved {len(transit_data['routes'])} transit routes to {output_path}")
print(f"Routes serving KU campus: {sum(1 for r in transit_data['routes'] if r['serves_ku'])}")