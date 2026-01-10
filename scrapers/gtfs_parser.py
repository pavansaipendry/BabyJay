import csv
import json
from datetime import datetime
from collections import defaultdict

GTFS_PATH = "/Users/pavansaipendry/Downloads/KU-LTS_GTFS/"

def parse_gtfs():
    """Parse complete GTFS data accurately"""
    
    print("Reading routes.txt...")
    routes = {}
    with open(GTFS_PATH + 'routes.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            routes[row['route_id']] = {
                'route_number': row['route_short_name'],
                'route_name': row['route_long_name'],
                'description': row['route_desc'].strip('"'),
                'color': '#' + row['route_color'],
                'url': row['route_url']
            }
    
    print("Reading calendar.txt...")
    service_patterns = {}
    with open(GTFS_PATH + 'calendar.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            service_patterns[row['service_id']] = {
                'monday': row['monday'] == '1',
                'tuesday': row['tuesday'] == '1',
                'wednesday': row['wednesday'] == '1',
                'thursday': row['thursday'] == '1',
                'friday': row['friday'] == '1',
                'saturday': row['saturday'] == '1',
                'sunday': row['sunday'] == '1',
                'start_date': row['start_date'],
                'end_date': row['end_date']
            }
    
    print("Reading trips.txt...")
    route_services = defaultdict(set)
    with open(GTFS_PATH + 'trips.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            route_services[row['route_id']].add(row['service_id'])
    
    print("Reading stops.txt...")
    stops = {}
    with open(GTFS_PATH + 'stops.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stops[row['stop_id']] = {
                'name': row['stop_name'],
                'lat': float(row['stop_lat']),
                'lon': float(row['stop_lon'])
            }
    
    print("Processing route data...")
    transit_data = {
        "last_updated": datetime.now().isoformat(),
        "data_source": "GTFS Feed - Lawrence Transit (Official)",
        "routes": [],
        "stops": stops
    }
    
    for route_id, route_info in routes.items():
        service_ids = route_services[route_id]
        
        # Determine operating days by combining all service patterns
        operates_days = {
            'monday': False,
            'tuesday': False,
            'wednesday': False,
            'thursday': False,
            'friday': False,
            'saturday': False,
            'sunday': False
        }
        
        for service_id in service_ids:
            if service_id in service_patterns:
                pattern = service_patterns[service_id]
                for day in operates_days:
                    if pattern[day]:
                        operates_days[day] = True
        
        # Categorize routes
        route_num = route_info['route_number']
        
        # Determine if route serves KU
        ku_routes = ['3', '4', '6', '7', '8', '11', '30', '34', '36', '38', '42', '43', '44', '45', '53']
        campus_only_routes = ['30', '34', '36', '38', '42', '43', '44', '45', '53']
        
        serves_ku = route_num in ku_routes
        campus_only = route_num in campus_only_routes
        
        # Build human-readable operating days
        weekdays = operates_days['monday'] and operates_days['friday']
        saturday = operates_days['saturday']
        sunday = operates_days['sunday']
        
        if weekdays and saturday and not sunday:
            operates = "Monday-Saturday"
        elif weekdays and not saturday and not sunday:
            operates = "Monday-Friday"
        elif operates_days['monday'] and operates_days['thursday'] and not operates_days['friday']:
            operates = "Monday-Thursday"
        elif all(operates_days.values()):
            operates = "Daily"
        else:
            days = []
            if operates_days['monday']: days.append('Mon')
            if operates_days['tuesday']: days.append('Tue')
            if operates_days['wednesday']: days.append('Wed')
            if operates_days['thursday']: days.append('Thu')
            if operates_days['friday']: days.append('Fri')
            if operates_days['saturday']: days.append('Sat')
            if operates_days['sunday']: days.append('Sun')
            operates = ', '.join(days) if days else "Special schedule"
        
        route_data = {
            'route_number': route_num,
            'route_name': route_info['route_name'],
            'description': route_info['description'],
            'operates': operates,
            'operates_days': operates_days,
            'serves_ku': serves_ku,
            'campus_only': campus_only,
            'color': route_info['color']
        }
        
        # Add special notes based on route number
        if route_num == '11':
            route_data['note'] = "Year-round service. Reduced schedule when KU not in session"
            route_data['popular_for_students'] = True
        elif route_num == '12':
            route_data['note'] = "Year-round service. Reduced schedule when KU not in session"
        elif route_num == '53':
            route_data['note'] = "Late night safety shuttle for KU campus"
            route_data['popular_for_students'] = True
            route_data['late_night'] = True
        elif route_num in ['42', '43']:
            route_data['popular_for_students'] = True
            route_data['note'] = "High frequency campus circulator"
        elif route_num == '4':
            route_data['popular_for_students'] = True
            route_data['note'] = "Connects West Lawrence to campus and downtown"
        elif route_num in ['30', '34', '36', '38', '44', '45']:
            route_data['note'] = "Operates only during Fall and Spring semesters while KU is in session"
        
        transit_data['routes'].append(route_data)
    
    # Sort routes numerically
    transit_data['routes'].sort(key=lambda x: (
        int(x['route_number']) if x['route_number'].isdigit() else 999,
        x['route_number']
    ))
    
    # Add system info
    transit_data['system_info'] = {
        'free_for_all': True,
        'free_through_year': 2025,
        'real_time_tracking': {
            'available': True,
            'apps': ['Transit App', 'Google Maps']
        },
        'service_hours': "Most routes: 6 AM - 8 PM Monday-Saturday",
        'customer_service': {
            'phone': '785-864-4644',
            'email': 'info@lawrencetransit.org',
            'website': 'lawrencetransit.org'
        },
        'most_useful_for_students': ['4', '11', '42', '43', '53'],
        'total_routes': len(transit_data['routes']),
        'total_stops': len(stops)
    }
    
    return transit_data

def main():
    print("\n" + "="*60)
    print("PARSING LAWRENCE TRANSIT GTFS DATA")
    print("="*60 + "\n")
    
    transit_data = parse_gtfs()
    
    # Save complete data
    output_path = "data/transit/routes.json"
    with open(output_path, 'w') as f:
        json.dump(transit_data, f, indent=2)
    
    print("\n" + "="*60)
    print("PARSING COMPLETE!")
    print("="*60)
    print(f"\nTotal routes: {len(transit_data['routes'])}")
    print(f"Routes serving KU: {sum(1 for r in transit_data['routes'] if r['serves_ku'])}")
    print(f"Campus-only routes: {sum(1 for r in transit_data['routes'] if r.get('campus_only'))}")
    print(f"Total bus stops: {len(transit_data['stops'])}")
    print(f"\nSaved to: {output_path}")
    
    print("\n" + "="*60)
    print("ROUTE SUMMARY")
    print("="*60)
    for route in transit_data['routes']:
        status = "CAMPUS" if route.get('campus_only') else ("KU" if route['serves_ku'] else "CITY")
        print(f"  Route {route['route_number']:>3} [{status:6}] {route['operates']:18} - {route['route_name'][:45]}")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE - NO ERRORS")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()