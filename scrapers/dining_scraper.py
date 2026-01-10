import json
from datetime import datetime

# KU Dining data collected from dining.ku.edu/hours
dining_data = {
    "last_updated": datetime.now().isoformat(),
    "locations": [
        {
            "id": 1,
            "name": "Mrs. E's",
            "building": "Lewis Hall",
            "address": "1532 Engel Rd",
            "type": "residential",
            "description": "All-you-care-to-eat dining hall in Daisy Hill",
            "hours": {
                "monday_friday": "7:00 AM - 7:30 PM",
                "saturday_sunday": "9:00 AM - 7:00 PM",
                "note": "Deli, Salad and Desserts only from 2:00-4:30 PM"
            },
            "coordinates": {
                "lat": 38.9543,
                "lng": -95.2535
            }
        },
        {
            "id": 2,
            "name": "South Dining Commons",
            "building": "Downs Hall",
            "address": "1517 W. 18th St",
            "type": "residential",
            "description": "All-you-care-to-eat dining hall",
            "hours": {
                "monday_friday": "7:00 AM - 7:30 PM",
                "saturday_sunday": "9:00 AM - 7:00 PM",
                "note": "Deli, Salad and Desserts only from 2:00-4:30 PM"
            },
            "coordinates": {
                "lat": 38.9565,
                "lng": -95.2601
            }
        },
        {
            "id": 3,
            "name": "North College Cafe",
            "building": "GSP Hall",
            "address": "500 W. 11th",
            "type": "residential",
            "description": "All-you-care-to-eat dining hall",
            "hours": {
                "monday_friday": "7:00 AM - 7:30 PM",
                "saturday_sunday": "9:00 AM - 7:00 PM",
                "note": "Deli, Salad and Desserts only from 2:00-4:30 PM"
            },
            "coordinates": {
                "lat": 38.9624,
                "lng": -95.2543
            }
        },
        {
            "id": 4,
            "name": "The Market",
            "building": "Kansas Union",
            "address": "1301 Jayhawk Blvd",
            "type": "retail",
            "description": "Grab-and-go market with various food options",
            "hours": {
                "monday_friday": "10:00 AM - 3:00 PM",
                "saturday_sunday": "Closed"
            },
            "coordinates": {
                "lat": 38.9575,
                "lng": -95.2456
            }
        },
        {
            "id": 5,
            "name": "Chick-fil-A",
            "building": "Kansas Union",
            "address": "1301 Jayhawk Blvd",
            "type": "retail",
            "description": "Chick-fil-A restaurant",
            "hours": {
                "monday_friday": "8:00 AM - 7:00 PM",
                "saturday": "10:00 AM - 6:00 PM",
                "sunday": "Closed"
            },
            "coordinates": {
                "lat": 38.9575,
                "lng": -95.2456
            }
        },
        {
            "id": 6,
            "name": "Wendy's",
            "building": "Kansas Union",
            "address": "1301 Jayhawk Blvd",
            "type": "retail",
            "description": "Wendy's restaurant",
            "hours": {
                "monday_friday": "10:00 AM - 8:00 PM",
                "saturday": "11:00 AM - 4:00 PM"
            },
            "coordinates": {
                "lat": 38.9575,
                "lng": -95.2456
            }
        },
        {
            "id": 7,
            "name": "The Underground",
            "building": "Wescoe Hall",
            "address": "1445 Jayhawk Blvd",
            "type": "retail",
            "description": "Quick service dining in Wescoe Hall",
            "hours": {
                "monday_friday": "10:00 AM - 3:00 PM",
                "saturday_sunday": "Closed"
            },
            "coordinates": {
                "lat": 38.9571,
                "lng": -95.2503
            }
        },
        {
            "id": 8,
            "name": "Courtside Cafe",
            "building": "DeBruce Center",
            "address": "1647 Naismith Dr",
            "type": "retail",
            "description": "Cafe in the basketball training facility",
            "hours": {
                "monday_thursday": "8:00 AM - 7:00 PM",
                "friday": "8:00 AM - 3:30 PM",
                "saturday_sunday": "Closed"
            },
            "coordinates": {
                "lat": 38.9545,
                "lng": -95.2562
            }
        },
        {
            "id": 9,
            "name": "The Studio",
            "building": "Hashinger Hall",
            "address": "1632 Engel Rd",
            "type": "retail",
            "description": "Late night dining option",
            "hours": {
                "monday_sunday": "4:30 PM - 11:00 PM"
            },
            "coordinates": {
                "lat": 38.9548,
                "lng": -95.2540
            }
        },
        {
            "id": 10,
            "name": "Shake Smart",
            "building": "Ambler Rec Center",
            "address": "1740 Watkins Center Dr",
            "type": "retail",
            "description": "Smoothies and healthy options at the rec center",
            "hours": {
                "monday_thursday": "8:00 AM - 9:00 PM",
                "friday": "8:00 AM - 5:00 PM",
                "saturday": "Closed",
                "sunday": "1:00 PM - 9:00 PM"
            },
            "coordinates": {
                "lat": 38.9553,
                "lng": -95.2615
            }
        }
    ]
}

# Save to JSON file
output_path = "data/dining/locations.json"
with open(output_path, 'w') as f:
    json.dump(dining_data, f, indent=2)

print(f"Saved {len(dining_data['locations'])} dining locations to {output_path}")