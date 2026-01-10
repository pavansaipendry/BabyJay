"""
KU Faculty Information Scraper
==============================
Purpose: Gather faculty information for BabyJay RAG system
Author: For educational use only

ETHICAL GUIDELINES FOLLOWED:
1. Rate limiting (2-3 second delays between requests)
2. Only scraping publicly available information
3. Respecting server resources
4. Educational/research purpose only

USAGE:
    python ku_faculty_scraper.py --department eecs
    python ku_faculty_scraper.py --department business
    python ku_faculty_scraper.py --all
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import random
import argparse
from urllib.parse import urljoin
import os
from urllib import robotparser
from urllib.parse import urlparse


DEPARTMENTS = {
    "eecs": {
        "name": "Electrical Engineering and Computer Science",
        "faculty_list_url": "https://eecs.ku.edu/people/list/faculty",
        "base_url": "https://eecs.ku.edu",
        "building": "Eaton Hall",
        "address": "2001 Eaton Hall, 1520 West 15th Street, Lawrence, KS 66045",
        "phone": "785-864-4620",
        "email": "eecs-info@ku.edu"
    },
    "business": {
        "name": "School of Business",
        "faculty_list_url": "https://business.ku.edu/people/list/faculty",
        "base_url": "https://business.ku.edu",
        "building": "Capitol Federal Hall",
        "address": "1654 Naismith Drive, Lawrence, KS 66045",
        "phone": "785-864-7500",
        "email": "bschoolinfo@ku.edu"
    },
    "psychology": {
    "name": "Department of Psychology",
    "faculty_list_url": "https://psychology.ku.edu/faculty",
    "base_url": "https://psychology.ku.edu",
    "building": "Fraser Hall",
    "address": "1415 Jayhawk Blvd., Lawrence, KS 66045",
    "phone": "785-864-4131",
    "email": "psychology@ku.edu"
},

"chemistry": {
    "name": "Department of Chemistry",
    "faculty_list_url": "https://chem.ku.edu/faculty",
    "base_url": "https://chem.ku.edu",
    "building": "Integrated Science Building (ISB)",
    "address": "1567 Irving Hill Road, Room 1140, Lawrence, KS 66045",
    "phone": "785-864-4670",
    "email": "chemistry@ku.edu"
},

"math": {
    "name": "Department of Mathematics",
    "faculty_list_url": "https://mathematics.ku.edu/faculty",
    "base_url": "https://mathematics.ku.edu",
    "building": "Snow Hall",
    "address": "405 Snow Hall, 1460 Jayhawk Blvd., Lawrence, KS 66045",
    "phone": "785-864-3651",
    "email": "math@ku.edu"

},

# "physics": {
#     "name": "Department of Physics and Astronomy",
#     "faculty_list_url": "https://physics.ku.edu/faculty",
#     "base_url": "https://physics.ku.edu",
#     "building": "Malott Hall",
#     "address": "Malott Hall, room 1082, 1251 Wescoe Hall Dr., Lawrence, KS 66045",
#     "phone": "785-864-4626",
#     "email": "physics@ku.edu"
# },

# "english": {
#     "name": "Department of English",
#     "faculty_list_url": "https://english.ku.edu/faculty",
#     "base_url": "https://english.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-4520",
#     "email": "english@ku.edu"
# },


    # =========================
    # Architecture & Design
    # =========================
#     "architecture_design": {
#         "name": "School of Architecture & Design",
#         "faculty_list_url": "https://arcd.ku.edu/all-faculty-staff",
#         "base_url": "https://arcd.ku.edu/",
#         "building": "Marvin Hall",
#         "address": "Marvin Hall, 1465 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-4281",
#         "email": "arcd@ku.edu"
#     },
#     "architecture": {
#         "name": "Architecture",
#         "faculty_list_url": "https://arcd.ku.edu/architecture-faculty",
#         "base_url": "https://arcd.ku.edu",
#         "building": "Marvin Hall",
#         "address": "Marvin Hall, 1465 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-4281",
#         "email": "arch@ku.edu"
#     },
#     "design": {
#         "name": "Design",
#         "faculty_list_url": "https://arcd.ku.edu/design-faculty",
#         "base_url": "https://design.ku.edu",
#         "building": "Marvin Hall, Room 200",
#         "address": "200 Marvin Hall, 1465 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-3390",
#         "email": "design@ku.edu"
#     },

#     # =========================
#     # Arts (School of the Arts in The College)
#     # =========================
#     "film_media_studies": {
#         "name": "Film & Media Studies",
#         "faculty_list_url": "https://film.ku.edu/faculty",
#         "base_url": "https://film.ku.edu",
#         "building": "Summerfield Hall, Suite 230",
#         "address": "1300 Sunnyside Ave., Suite 230, Lawrence, KS 66045",
#         "phone": "785-864-1340",
#         "email": "film@ku.edu"
#     },
#     "theatre_dance": {
#         "name": "Theatre & Dance",
#         "faculty_list_url": "https://theatredance.ku.edu/faculty-staff",
#         "base_url": "https://theatredance.ku.edu",
#         "building": "Murphy Hall, Room 356",
#         "address": "1530 Naismith Drive, Lawrence, KS 66045",
#         "phone": "785-864-3511",
#         "email": "kuthr@ku.edu"
#     },
#     "visual_art": {
#         "name": "Visual Art",
#         "faculty_list_url": "https://art.ku.edu/faculty-and-staff",
#         "base_url": "https://art.ku.edu",
#         "building": "Chalmers Hall, Room 300",
#         "address": "300 Chalmers Hall, 1467 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-4042",
#         "email": "visualart@ku.edu"
#     },

#     # =========================
#     # Education & Human Sciences
#     # =========================
#     "education_human_sciences": {
#         "name": "School of Education & Human Sciences",
#         "faculty_list_url": "https://soehs.ku.edu/leadership-staff",
#         "base_url": "https://soehs.ku.edu",
#         "building": "Joseph R. Pearson Hall, Rm 221",
#         "address": "1122 West Campus Rd. Lawrence, KS 66045-3101",
#         "phone": "785-864-3726",
#         "email": "soehs@ku.edu" 
#     },
#     "curriculum_teaching": {
#         "name": "Curriculum & Teaching",
#         "faculty_list_url": "https://ct.ku.edu/faculty-staff",
#         "base_url": "https://ct.ku.edu",
#         "building": "Joseph R. Pearson Hall, Rm. 321",
#         "address": " 1122 West Campus Rd. Lawrence, KS 66045-3101",
#         "phone": "785-864-4435",
#         "email": "ctdepartment@ku.edu"
#     },
#     "educational_leadership_policy_studies": {
#         "name": "Educational Leadership & Policy Studies",
#         "faculty_list_url": "https://elps.ku.edu/faculty-staff",
#         "base_url": "https://elps.ku.edu",
#         "building": "Joseph R. Pearson Hall, Rm. 321",
#         "address": "1122 West Campus Rd. Lawrence, KS 66045-3101",
#         "phone": "785-864-4458",
#         "email": "elps@ku.edu"
#     },

#     "educational_psychology": {
#     "name": "Educational Psychology",
#     "faculty_list_url": "https://epsy.ku.edu/faculty-staff",
#     "base_url": "https://epsy.ku.edu",
#     "building": "Joseph R. Pearson Hall",
#     "address": "Joseph R. Pearson Hall, Room 621, 1122 West Campus Rd., Lawrence, KS 66045-3101",
#     "phone": "785-864-3931",
#     "email": "epsy@ku.edu"
# },
# "health_sport_exercise_sciences": {
#     "name": "Health, Sport & Exercise Sciences",
#     "faculty_list_url": "https://hses.ku.edu/faculty-staff",
#     "base_url": "https://hses.ku.edu",
#     "building": "Robinson Center",
#     "address": "Robinson Center, Room 161, 1301 Sunnyside Ave., Lawrence, KS 66045",
#     "phone": "785-864-0784",
#     "email": "hses@ku.edu"
# },
# "special_education": {
#     "name": "Special Education",
#     "faculty_list_url": "https://specialedu.ku.edu/faculty-staff",
#     "base_url": "https://specialedu.ku.edu",
#     "building": "Joseph R. Pearson Hall",
#     "address": "Joseph R. Pearson Hall, Room 521, 1122 West Campus Rd., Lawrence, KS 66045-3101",
#     "phone": "785-864-4954",
#     "email": "specialedu@ku.edu"
# },

# "engineering": {
#     "name": "School of Engineering",
#     "faculty_list_url": "https://engr.ku.edu/people",
#     "base_url": "https://engr.ku.edu",
#     "building": "Eaton Hall",
#     "address": "1 Eaton Hall, 1520 W. 15th Street, Lawrence, KS 66045-7609",
#     "phone": "785-864-3881",
#     "email": "kuengr@ku.edu"
# },
# "aerospace_engineering": {
#     "name": "Aerospace Engineering",
#     "faculty_list_url": "https://ae.ku.edu/ae-faculty",
#     "base_url": "https://ae.ku.edu",
#     "building": "Learned Hall",
#     "address": "2120 Learned Hall, 1530 W. 15th Street, Lawrence, KS 66045-7609",
#     "phone": "785-864-2960",
#     "email": "aero@ku.edu"
# },
    "bioengineering": {
        "name": "Bioengineering",
        "faculty_list_url": "https://bioengineering.ku.edu/people",
        "base_url": "https://bioengineering.ku.edu",
        "building": "Eaton Hall",
        "address": "1 Eaton Hall, 1520 W. 15th Street, Lawrence, KS 66045",
        "phone": "785-864-4931",
        "email": "bioe@ku.edu"
},
    "chemical_petroleum_engineering": {
        "name": "Chemical & Petroleum Engineering",
        "faculty_list_url": "https://cpe.ku.edu/faculty",
        "base_url": "https://cpe.ku.edu",
        "building": "Slawson Hall",
        "address": "Slawson Hall, 1536 W. 15th Street, Lawrence, KS 66045",
        "phone": "785-864-4965",
        "email": "cpe@ku.edu"
},

"civil_environmental_architectural_engineering": { 
    "name": "Civil, Environmental & Architectural Engineering",
    "faculty_list_url": "https://ceae.ku.edu/faculty",
    "base_url": "https://ceae.ku.edu",
    "building": "Learned Hall",
    "address": "2150 Learned Hall, 1530 W. 15th Street, Lawrence, KS 66045-7609",
    "phone": "785-864-3827",
    "email": "ceae@ku.edu"
},
# "engineering_physics": {
#     "name": "Engineering Physics",
#     "faculty_list_url": "https://ephx.ku.edu/people/list/faculty",
#     "base_url": "https://ephx.ku.edu",
#     "building": "Malott Hall",
#     "address": "1082 Malott Hall, 1251 Wescoe Hall Dr., Lawrence, KS 66045",
#     "phone": "785-864-4626",
#     "email": "ephx@ku.edu"
# },
"mechanical_engineering": {
    "name": "Mechanical Engineering",
    "faculty_list_url": "https://me.ku.edu/faculty",
    "base_url": "https://me.ku.edu",
    "building": "Learned Hall",
    "address": "3138 Learned Hall, 1530 W. 15th Street, Lawrence, KS 66045-7609",
    "phone": "785-864-3181",
    "email": "me@ku.edu"
},

# "project_management": {
#     "name": "Project Management",
#     "faculty_list_url": "https://sps.ku.edu/programs/project-management/people",
#     "base_url": "https://sps.ku.edu/programs/project-management",
#     "building": "Edwards Campus",
#     "address": "12600 Quivira Road Overland Park, KS 66213",
#     "phone": "913-897-8400",
#     "email": "professionalstudies@ku.edu"
# },

#         # =========================
#     # Journalism, Law, Music
#     # =========================
#     "journalism": {
#         "name": "Journalism & Mass Communications",
#         "faculty_list_url": "https://journalism.ku.edu/faculty",
#         "base_url": "https://journalism.ku.edu",
#         "building": "Stauffer-Flint Hall",
#         "address": "Stauffer-Flint Hall, 1435 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-4755",
#         "email": "jschool@ku.edu"
#     },
#     "law": {
#         "name": "School of Law",
#         "faculty_list_url": "https://law.ku.edu/faculty",
#         "base_url": "https://law.ku.edu",
#         "building": "Green Hall",
#         "address": "Green Hall, 1535 W. 15th Street, Lawrence, KS 66045",
#         "phone": "785-864-4550",
#         "email": "admitlaw@ku.edu"
#     },
#     "music": {
#         "name": "School of Music",
#         "faculty_list_url": "https://music.ku.edu/directory",
#         "base_url": "https://music.ku.edu",
#         "building": "Murphy Hall",
#         "address": "Murphy Hall, 1530 Naismith Drive, Lawrence, KS 66045",
#         "phone": "785-864-3436",
#         "email": "music@ku.edu"
#     },

#     # =========================
#     # Liberal Arts & Sciences (sub-departments)
#     # =========================
#     "african_african_american_studies": {
#         "name": "African & African-American Studies",
#         "faculty_list_url": "https://afs.ku.edu/faculty",
#         "base_url": "https://afs.ku.edu",
#         "building": "Bailey Hall",
#         "address": "Bailey Hall, 1450 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-3054",
#         "email": "afs@ku.edu"
#     },
#     "american_studies": {
#         "name": "American Studies",
#         "faculty_list_url": "https://americanstudies.ku.edu/people/list",
#         "base_url": "https://americanstudies.ku.edu",
#         "building": "Bailey Hall",
#         "address": "Bailey Hall, 1440 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-4011",
#         "email": "amerst@ku.edu"
#     },
#     "anthropology": {
#         "name": "Anthropology",
#         "faculty_list_url": "https://anthropology.ku.edu/faculty",
#         "base_url": "https://anthropology.ku.edu",
#         "building": "Fraser Hall",
#         "address": "622 Fraser Hall, 1415 Jayhawk Blvd., Lawrence, KS 66045",
#         "phone": "785-864-2630",
#         "email": "kuanthro@ku.edu"
#     },
#     "applied_behavioral_science": {
#         "name": "Applied Behavioral Science",
#         "faculty_list_url": "https://absc.ku.edu/faculty",
#         "base_url": "https://absc.ku.edu",
#         "building": "Dole Human Development Center",
#         "address": "4001 Dole Human Development Center, 1000 Sunnyside Ave., Lawrence, KS 66045",
#         "phone": "785-864-4840",
#         "email": "absc@ku.edu"
#     },
#     "biology": {
#         "name": "Biology",
#         "faculty_list_url": "https://kuub.ku.edu/faculty",
#         "base_url": "https://kuub.ku.edu",
#         "building": "Haworth Hall",
#         "address": "2045 Haworth Hall, 1200 Sunnyside Ave., Lawrence, KS 66045",
#         "phone": "785-864-4301",
#         "email": "kuub@ku.edu"
#     },
#     "child_language": {
#         "name": "Child Language",
#         "faculty_list_url": "https://cldp.ku.edu/affiliated-faculty",
#         "base_url": "https://cldp.ku.edu",
#         "building": "Dole Human Development Center",
#         "address": "3031 Dole Human Development Center, 1000 Sunnyside Ave., Lawrence, KS 66045",
#         "phone": "785-864-4570",
#         "email": "childlang@ku.edu"
#     },

#     "classics": {
#     "name": "Classics",
#     "faculty_list_url": "https://classics.ku.edu/faculty",
#     "base_url": "https://classics.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "1021 Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045-7590",
#     "phone": "785-864-3153",
#     "email": "classics@ku.edu"
# },
# "clinical_child_psychology": {
#     "name": "Clinical Child Psychology",
#     "faculty_list_url": "https://ccpp.ku.edu/core-faculty",
#     "base_url": "https://ccpp.ku.edu",
#     "building": "Learned Hall",
#     "address": "2111 Learned Hall, 1530 W. 15th St., Lawrence, KS 66045",
#     "phone": "785-864-9803",
#     "email": "ccpp@ku.edu"
# },
# "communication_studies": {
#     "name": "Communication Studies",
#     "faculty_list_url": "https://coms.ku.edu/faculty",
#     "base_url": "https://coms.ku.edu",
#     "building": "Bailey Hall",
#     "address": "Bailey Hall, Room 102, 1440 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-9878",
#     "email": "coms@ku.edu"
# },
# "computational_biology": {
#     "name": "Computational Biology",
#     "faculty_list_url": "https://compbio.ku.edu/faculty",
#     "base_url": "https://compbio.ku.edu",
#     "building": "Haworth Hall",
#     "address": "1200 Sunnyside Ave, 4049 Haworth Hall, Lawrence, KS 66045",
#     "phone": "785-864-7333",
#     "email": "compbio@ku.edu"
# },
# "east_asian_languages_cultures": {
#     "name": "East Asian Languages & Cultures",
#     "faculty_list_url": "https://ealc.ku.edu/faculty",
#     "base_url": "https://ealc.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "1445 Jayhawk Blvd., Room 2053 Wescoe Hall, Lawrence, KS 66045",
#     "phone": "785-864-9250",
#     "email": "ealc@ku.edu"
# },
# "ecology_evolutionary_biology": {
#     "name": "Ecology & Evolutionary Biology",
#     "faculty_list_url": "https://eeb.ku.edu/faculty",
#     "base_url": "https://eeb.ku.edu",
#     "building": "Haworth Hall",
#     "address": "1200 Sunnyside Ave., 2041 Haworth Hall, Lawrence, KS 66045",
#     "phone": "785-864-5884",
#     "email": "eeb@ku.edu"
# },
# "economics": {
#     "name": "Economics",
#     "faculty_list_url": "https://economics.ku.edu/facultyall",
#     "base_url": "https://economics.ku.edu",
#     "building": "Snow Hall",
#     "address": "1460 Jayhawk Blvd., 415 Snow Hall, Lawrence, KS 66045",
#     "phone": "785-864-2861",
#     "email": "econ@ku.edu"
# },
# "environmental_studies": {
#     "name": "Environmental Studies",
#     "faculty_list_url": "https://esp.ku.edu/faculty",
#     "base_url": "https://esp.ku.edu",
#     "building": "Lindley Hall",
#     "address": "1475 Jayhawk Blvd., 213 Lindley Hall, Lawrence, KS 66045",
#     "phone": "785-864-4973",
#     "email": "envstudies@ku.edu"
# },
# "french_francophone_italian": {
#     "name": "French, Francophone & Italian Studies",
#     "faculty_list_url": "https://frenchitalian.ku.edu/faculty",
#     "base_url": "https://frenchitalian.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "1445 Jayhawk Blvd., Room 2103 Wescoe Hall, Lawrence, KS 66045",
#     "phone": "785-864-4056",
#     "email": "frenchitalian@ku.edu"
# },
# "geography_atmospheric_sciences": {
#     "name": "Geography & Atmospheric Sciences",
#     "faculty_list_url": "https://geog.ku.edu/faculty",
#     "base_url": "https://geog.ku.edu",
#     "building": "Lindley Hall",
#     "address": "1475 Jayhawk Blvd., 213 Lindley Hall, Lawrence, KS 66045",
#     "phone": "785-864-5143",
#     "email": "geog@ku.edu"
# },
# "geology": {
#     "name": "Geology",
#     "faculty_list_url": "https://geo.ku.edu/faculty",
#     "base_url": "https://geo.ku.edu",
#     "building": "Lindley Hall",
#     "address": "1414 Naismith Drive, Room 254, Lawrence, KS 66045",
#     "phone": "785-864-4974",
#     "email": "geology@ku.edu"
# },
# "global_international_studies": {
#     "name": "Global & International Studies",
#     "faculty_list_url": "https://global.ku.edu/center-faculty-staff",
#     "base_url": "https://global.ku.edu",
#     "building": "Bailey Hall",
#     "address": "Bailey Hall, Room 318, 1440 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-3748",
#     "email": "global@ku.edu"
# },

# "history": {
# "name": "History",
# "faculty_list_url": "https://history.ku.edu/faculty",
# "base_url": "https://history.ku.edu",
# "building": "Wescoe Hall",
# "address": "3650 Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
# "phone": "785-864-3569",
# "email": "historyhr@ku.edu"
# },

# "history_of_art": {
#     "name": "History of Art",
#     "faculty_list_url": "https://arthistory.ku.edu/faculty",
#     "base_url": "https://arthistory.ku.edu",
#     "building": "Spencer Museum of Art",
#     "address": "Spencer Museum of Art, 1301 Mississippi St., Lawrence, KS 66045",
#     "phone": "785-864-4713",
#     "email": "arthistory@ku.edu"
# },
# "indigenous_studies": {
#     "name": "Indigenous Studies",
#     "faculty_list_url": "https://indigenous.ku.edu/faculty-affiliate-faculty",
#     "base_url": "https://indigenous.ku.edu",
#     "building": "Bailey Hall",
#     "address": "310 Bailey Hall, 1440 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-2660",
#     "email": "indigenous@ku.edu"
# },
# "leadership_studies": {
#     "name": "Institute for Leadership Studies",
#     "faculty_list_url": "https://ils.ku.edu/academic-staff",
#     "base_url": "https://ils.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "1130A Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-4225",
#     "email": "leadershipstudies@ku.edu"
# },
# "jewish_studies": {
#     "name": "Jewish Studies",
#     "faculty_list_url": "https://kujewishstudies.ku.edu/people",
#     "base_url": "https://jewishstudies.ku.edu",
#     "building": "Bailey Hall",
#     "address": "309 Bailey Hall, 1440 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-4664",
#     "email": "jewishstudies@ku.edu"
# },
# "linguistics": {
#     "name": "Linguistics",
#     "faculty_list_url": "https://linguistics.ku.edu/faculty",
#     "base_url": "https://linguistics.ku.edu",
#     "building": "Blake Hall",
#     "address": "427 Blake Hall, 1541 Lilac Ln., Lawrence, KS 66045",
#     "phone": "785-864-3450",
#     "email": "linguistics@ku.edu"
# },
# "molecular_biosciences": {
#     "name": "Molecular Biosciences",
#     "faculty_list_url": "https://molecularbiosciences.ku.edu/faculty",
#     "base_url": "https://molecularbiosciences.ku.edu",
#     "building": "Haworth Hall",
#     "address": "5034 Haworth Hall, 1200 Sunnyside Ave., Lawrence, KS 66045",
#     "phone": "785-864-4142",
#     "email": "molecularbiosciences@ku.edu"
# },
# "museum_studies": {
#     "name": "Museum Studies",
#     "faculty_list_url": "https://museumstudies.ku.edu/faculty",
#     "base_url": "https://museumstudies.ku.edu",
#     "building": "Bailey Hall",
#     "address": "435 Bailey Hall, 1440 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-9240",
#     "email": "museumstudies@ku.edu"
# },
# "philosophy": {
#     "name": "Philosophy",
#     "faculty_list_url": "https://philosophy.ku.edu/faculty",
#     "base_url": "https://philosophy.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "3090 Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-3976",
#     "email": "philosophy@ku.edu"
# },
# "political_science": {
#     "name": "Political Science",
#     "faculty_list_url": "https://kups.ku.edu/faculty",
#     "base_url": "https://kups.ku.edu",
#     "building": "Blake Hall",
#     "address": "504 Blake Hall, 1541 Lilac Ln., Lawrence, KS 66045",
#     "phone": "785-864-3523",
#     "email": "kupolsci@ku.edu"
# },
# "religious_studies": {
#     "name": "Religious Studies",
#     "faculty_list_url": "https://religiousstudies.ku.edu/faculty",
#     "base_url": "https://religiousstudies.ku.edu",
#     "building": "Smith Hall",
#     "address": "203 Smith Hall, 1300 Oread Ave., Lawrence, KS 66045",
#     "phone": "785-864-4663",
#     "email": "religiousstudies@ku.edu"
# },

#     "slavic_german_eurasian": {
#     "name": "Slavic, German & Eurasian Studies",
#     "faculty_list_url": "https://sges.ku.edu/faculty",
#     "base_url": "https://sges.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "2080 Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-9250",
#     "email": "sges@ku.edu"
# },
# "sociology": {
#     "name": "Sociology",
#     "faculty_list_url": "https://sociology.ku.edu/faculty",
#     "base_url": "https://sociology.ku.edu",
#     "building": "Fraser Hall",
#     "address": "716 Fraser Hall, 1415 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-9426",
#     "email": "sociology@ku.edu"
# },
# "spanish_portuguese": {
#     "name": "Spanish & Portuguese",
#     "faculty_list_url": "https://spanport.ku.edu/faculty",
#     "base_url": "https://spanport.ku.edu",
#     "building": "Wescoe Hall",
#     "address": "2603 Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-3851",
#     "email": "spanport@ku.edu"
# },
# "speech_language_hearing": {
#     "name": "Speech-Language-Hearing: Sciences & Disorders",
#     "faculty_list_url": "https://splh.ku.edu/department-faculty",
#     "base_url": "https://splh.ku.edu",
#     "building": "Dole Human Development Center",
#     "address": "3001 Dole Human Development Center, 1000 Sunnyside Ave, Lawrence, KS 66045",
#     "phone": "785-864-0630",
#     "email": "splh@ku.edu"
# },
# "women_gender_sexuality_studies": {
#     "name": "Women, Gender, and Sexuality Studies",
#     "faculty_list_url": "https://wgss.ku.edu/faculty-directory",
#     "base_url": "https://wgss.ku.edu",
#     "building": "Blake Hall",
#     "address": "318 Blake Hall, 1541 Lilac Lane, Lawrence, KS 66045",
#     "phone": "785-864-2310",
#     "email": "wgss@ku.edu"
# },


# # =========================
# # Pharmacy + subunits
# # =========================
# "pharmacy": {
#     "name": "School of Pharmacy",
#     "faculty_list_url": "https://pharmacy.ku.edu/people/list/faculty",
#     "base_url": "https://pharmacy.ku.edu",
#     "building": "School of Pharmacy building",
#     "address": "2010 Becker Drive, Lawrence, KS 66047",
#     "phone": "785-864-3591",
#     "email": "pharmacy@ku.edu"
# },
# "medicinal_chemistry": {
#     "name": "Medicinal Chemistry",
#     "faculty_list_url": "https://medchem.ku.edu/people/list/faculty",
#     "base_url": "https://medchem.ku.edu",
#     "building": "Shankel Structural Biology Center",
#     "address": "2034 Becker Drive, Lawrence, KS 66047",
#     "phone": "785-864-4495",
#     "email": "medchem@ku.edu"
# },
# "neurosciences": {
#     "name": "Neurosciences",
#     "faculty_list_url": "https://pharmtox.ku.edu/neuroscience-faculty",
#     "base_url": "https://neuroscience.ku.edu",
#     "building": "Malott Hall",
#     "address": "Malott Hall, Room 5064, 1251 Wescoe Hall Drive, Lawrence, KS 66045",
#     "phone": "785-864-4002",
#     "email": "pharmtox@ku.edu"
# },
# "pharmaceutical_chemistry": {
#     "name": "Pharmaceutical Chemistry",
#     "faculty_list_url": "https://pharmchem.ku.edu/people/list/faculty",
#     "base_url": "https://pharmchem.ku.edu",
#     "building": "Simons Research Laboratories",
#     "address": "2095 Constant Avenue, Lawrence, KS 66047",
#     "phone": "785-864-4822",
#     "email": "pharmchem@ku.edu"
# },
# "pharmacology_toxicology": {
#     "name": "Pharmacology & Toxicology",
#     "faculty_list_url": "https://pharmtox.ku.edu/people/list/faculty",
#     "base_url": "https://pharmtox.ku.edu",
#     "building": "Malott Hall",
#     "address": "Malott Hall, Room 5064, 1251 Wescoe Hall Drive, Lawrence, KS 66045",
#     "phone": "785-864-4002",
#     "email": "pharmtox@ku.edu"
# },
# "pharmacy_practice": {
#     "name": "Pharmacy Practice",
#     "faculty_list_url": "https://pharmpractice.ku.edu/people/list/faculty",
#     "base_url": "https://pharmpractice.ku.edu",
#     "building": "School of Pharmacy building",
#     "address": "2010 Becker Drive, Lawrence, KS 66047",
#     "phone": "785-864-3591",
#     "email": "pharmacy@ku.edu"
# },

# # =========================
# # Professional Studies & Public Affairs
# # =========================
# "professional_studies": {
#     "name": "School of Professional Studies",
#     "faculty_list_url": "https://sps.ku.edu/programs/prof-studies/people",
#     "base_url": "https://sps.ku.edu",
#     "building": "KU Edwards Campus",
#     "address": "12600 Quivira Road, Overland Park, KS 66213",
#     "phone": "913-897-8400",
#     "email": "professionalstudies@ku.edu"
# },
# "public_affairs_administration": {
#     "name": "Public Affairs & Administration",
#     "faculty_list_url": "https://spaa.ku.edu/people/faculty",
#     "base_url": "https://spaa.ku.edu",
#     "building": "Wescoe Hall, Room 4060",
#     "address": "Wescoe Hall, 1445 Jayhawk Blvd., Lawrence, KS 66045",
#     "phone": "785-864-3527",
#     "email": "padept@ku.edu"
# },


# "social_welfare": {
#     "name": "School of Social Welfare",
#     "faculty_list_url": "https://socwel.ku.edu/people/faculty",
#     "base_url": "https://socwel.ku.edu",
#     "building": "Green Hall",
#     "address": "1535 West 15th Street Lawrence, KS 66045",
#     "phone": "785-864-4720",
#     "email": "socwel@ku.edu"
# }
}


# Rate limiting settings (BE RESPECTFUL TO THE SERVER!)
MIN_DELAY = 2  # Minimum seconds between requests
MAX_DELAY = 5  # Maximum seconds between requests

# User agent to identify ourselves
HEADERS = {
    "User-Agent": "KU-BabyJay-Research-Bot/1.0 (Educational Project; Contact: pavansaipendry2002@ku.edu)"
}
ROBOTS_CACHE = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def polite_delay():
    """
    Wait a random time between requests to be respectful to the server.
    This prevents overwhelming the server (anti-DDoS measure).
    """
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"    Waiting {delay:.1f} seconds.")
    time.sleep(delay)


def safe_request(url, max_retries=3):
    """
    Make a request with error handling and retries.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"    ⚠️ Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retry
    return None


def extract_text_safe(element):
    """Safely extract text from a BeautifulSoup element."""
    if element:
        return element.get_text(strip=True)
    return ""


# ============================================================================
# SCRAPING FUNCTIONS
# ============================================================================

def get_faculty_list_urls(department_key):
    """
    Step 1: Get the list of all faculty profile URLs from the department page.
    
    This function:
    1. Loads the department's faculty list page
    2. Finds all links to individual faculty profiles
    3. Returns a list of URLs
    """
    dept = DEPARTMENTS[department_key]
    print(f"\n Getting faculty list from: {dept['faculty_list_url']}")
    
    response = safe_request(dept['faculty_list_url'])
    if not response:
        print("Failed to load faculty list page")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all faculty profile links
    # KU sites typically have links like /people/firstname-lastname
    faculty_urls = []
    
    # Look for links containing '/people/' in href
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Filter for faculty profile links
        if '/people/' in href and href != '/people/' and '/list' not in href:
            # Convert relative URLs to absolute
            full_url = urljoin(dept['base_url'], href)
            
            # Avoid duplicates
            if full_url not in faculty_urls:
                faculty_urls.append(full_url)
    
    print(f"Found {len(faculty_urls)} faculty profile links")
    return faculty_urls


def scrape_faculty_profile(url, department_key):
    """
    Step 2: Scrape an individual faculty profile page.
    
    This extracts:
    - Name
    - Title
    - Office location
    - Phone number
    - Email
    - Personal website
    - Research interests
    - Biography (research description)
    """
    print(f"\n👤 Scraping: {url}")
    
    response = safe_request(url)
    if not response:
        print(f"Failed to load profile")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    faculty = {
        "name": "",
        "title": "",
        "office": "",
        "phone": "",
        "email": "",
        "website": "",
        "research_interests": [],
        "biography": "",
        "profile_url": url
    }
    
    # ---- Extract Name ----
    # Usually in an h1 tag
    name_tag = soup.find('h1')
    if name_tag:
        faculty['name'] = extract_text_safe(name_tag)
        print(f"Name: {faculty['name']}")
    
    # ---- Extract Title ----
    # Often in a list or specific div
    title_elements = soup.find_all(['li', 'p', 'div'], class_=lambda x: x and 'title' in x.lower() if x else False)
    if not title_elements:
        # Try finding in the content area after h1
        main_content = soup.find('main') or soup.find('article') or soup
        title_list = main_content.find('ul')
        if title_list:
            titles = [extract_text_safe(li) for li in title_list.find_all('li')]
            faculty['title'] = '; '.join(titles[:3])  # Take first 3 titles max
    else:
        faculty['title'] = extract_text_safe(title_elements[0])
    
    # ---- Extract Contact Info ----
    # Look for email links
    email_link = soup.find('a', href=lambda x: x and 'mailto:' in x if x else False)
    if email_link:
        faculty['email'] = email_link['href'].replace('mailto:', '').strip()
        print(f"Email: {faculty['email']}")
    
    # Look for phone links
    phone_link = soup.find('a', href=lambda x: x and 'tel:' in x if x else False)
    if phone_link:
        faculty['phone'] = phone_link['href'].replace('tel:', '').replace('+1-', '').strip()
        print(f"Phone: {faculty['phone']}")
    
        # ---- Extract Office ----
    # Goal: get the *personal* office (e.g., "331 Strong"), not the dept footer ("405 Snow Hall").
    faculty['office'] = ""
    office = ""

    # Strategy 1: KU-style "Contact Info" block (Math and many others)
    contact_heading = None
    for h in soup.find_all(['h2', 'h3', 'h4']):
        if 'contact info' in h.get_text(strip=True).lower():
            contact_heading = h
            break

    if contact_heading:
        # Collect lines under "Contact Info" until the next heading
        sib = contact_heading.find_next_sibling()
        lines = []
        while sib and sib.name not in ['h2', 'h3', 'h4']:
            text = sib.get_text("\n", strip=True)
            if text:
                lines.extend([ln.strip() for ln in text.split("\n") if ln.strip()])
            sib = sib.find_next_sibling()

        # Heuristic: skip email/phone; pick the line that looks like an office
        for ln in lines:
            lower = ln.lower()
            if '@' in ln:
                continue  # email
            if any(ch.isdigit() for ch in ln) and any(
                kw in lower
                for kw in [
                    "hall", "strong", "fraser", "eaton",
                    "capitol federal", "wescoe", "blake",
                    "malott", "naismith", "snow"
                ]
            ):
                office = ln
                break

    # Strategy 2: fallback pattern search on the page, but avoid department footer
    if not office:
        page_text = soup.get_text("\n", strip=True)
        import re
        pattern = r'(\d+[A-Z]?\s+(Strong|Strong Hall|Fraser Hall|Eaton Hall|Capitol Federal Hall|Wescoe Hall|Blake Hall|Malott Hall|Snow Hall))'
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            candidate = match.group(0).strip()
            # Filter out the generic department address lines
            bad_lines = {
                "405 snow hall",
                "1460 jayhawk blvd.",
                "1460 jayhawk blvd",
            }
            if candidate.lower() not in bad_lines:
                office = candidate

    faculty['office'] = office
    if faculty['office']:
        print(f"    🏢 Office: {faculty['office']}")

    
    # ---- Extract Personal Website ----
    # Look for "Personal Links" section or external links
    personal_links = soup.find_all('a', href=lambda x: x and ('people.eecs.ku.edu' in x or 
                                                               'ittc.ku.edu' in x or
                                                               'faculty' in x.lower()) if x else False)
    for link in personal_links:
        href = link.get('href', '')
        if href and 'ku.edu' in href and href != url:
            faculty['website'] = href
            break
    
    # ---- Extract Research Interests ----
    # Usually under a "Research" heading
    research_section = None
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        if 'research' in heading.get_text().lower():
            research_section = heading.find_next(['ul', 'div', 'p'])
            break
    
    if research_section:
        if research_section.name == 'ul':
            faculty['research_interests'] = [extract_text_safe(li) for li in research_section.find_all('li')]
        else:
            text = extract_text_safe(research_section)
            # Split by common delimiters
            if ',' in text:
                faculty['research_interests'] = [r.strip() for r in text.split(',')]
            elif ';' in text:
                faculty['research_interests'] = [r.strip() for r in text.split(';')]
            else:
                faculty['research_interests'] = [text]
    
    print(f"Research areas: {len(faculty['research_interests'])} found")
    
    # ---- Extract Biography ----
    # Usually under a "Biography" heading or in main content
    bio_section = None
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        heading_text = heading.get_text().lower()
        if 'biography' in heading_text or 'about' in heading_text:
            # Get all following paragraphs until next heading
            bio_parts = []
            sibling = heading.find_next_sibling()
            while sibling and sibling.name not in ['h2', 'h3', 'h4']:
                if sibling.name == 'p':
                    bio_parts.append(extract_text_safe(sibling))
                sibling = sibling.find_next_sibling()
            faculty['biography'] = ' '.join(bio_parts)
            break
    
    # If no biography section, try to get from main content
    if not faculty['biography']:
        # Look for longer text blocks that might be biographical
        for p in soup.find_all('p'):
            text = extract_text_safe(p)
            # Biographical text is usually longer and talks about the person
            if len(text) > 200 and (faculty['name'].split()[0] in text or 'research' in text.lower()):
                faculty['biography'] = text
                break
    
    if faculty['biography']:
        print(f" Biography: {len(faculty['biography'])} characters")
    
    return faculty


def scrape_department(department_key):
    """
    Main function to scrape all faculty from a department.
    """
    if department_key not in DEPARTMENTS:
        print(f"Unknown department: {department_key}")
        print(f"Available departments: {', '.join(DEPARTMENTS.keys())}")
        return None
    
    dept = DEPARTMENTS[department_key]
    print(f"\n{'='*60}")
    print(f"SCRAPING: {dept['name']}")
    print(f"{'='*60}")
    
    # Get all faculty URLs
    faculty_urls = get_faculty_list_urls(department_key)
    
    if not faculty_urls:
        print("No faculty URLs found")
        return None
    
    # Scrape each faculty profile
    all_faculty = []
    total = len(faculty_urls)
    
    for i, url in enumerate(faculty_urls, 1):
        print(f"\n[{i}/{total}] ", end="")
        
        faculty = scrape_faculty_profile(url, department_key)
        if faculty and faculty['name']:
            all_faculty.append(faculty)
        
        # Be polite - wait between requests
        if i < total:  # Don't wait after the last one
            polite_delay()
    
    # Create the department data structure
    department_data = {
        "name": dept['name'],
        "building": dept['building'],
        "address": dept['address'],
        "phone": dept['phone'],
        "email": dept['email'],
        "website": dept.get('base_url', ''),
        "faculty_count": len(all_faculty),
        "faculty": all_faculty
    }
    
    print(f"Successfully scraped {len(all_faculty)} faculty members")
    return department_data


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_to_json(data, filename):
    """Save scraped data to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scrape KU faculty information (ethically!)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python ku_faculty_scraper.py --department eecs
            python ku_faculty_scraper.py --department business
            python ku_faculty_scraper.py --all
            
        Available departments:
            eecs      - Electrical Engineering and Computer Science
            business  - School of Business
            psychology - Department of Psychology
            chemistry - Department of Chemistry
            math      - Department of Mathematics
            physics   - Department of Physics and Astronomy
            english   - Department of English
        """
    )
    
    parser.add_argument('--department', '-d', 
                        help='Department to scrape (see list below)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Scrape all departments')
    parser.add_argument('--output-dir', default='scraped_data',
                        help='Output directory (default: scraped_data)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which departments to scrape
    if args.all:
        departments_to_scrape = list(DEPARTMENTS.keys())
    elif args.department:
        departments_to_scrape = [args.department]
    else:
        parser.print_help()
        return
    
    # Scrape each department
    all_data = {}
    
    for dept_key in departments_to_scrape:
        data = scrape_department(dept_key)
        
        if data:
            all_data[dept_key] = data
            
            # Save individual department files (JSON only)
            base_filename = f"{args.output_dir}/{dept_key}_faculty"
            save_to_json(data, f"{base_filename}.json")
        
        # Extra delay between departments
        if dept_key != departments_to_scrape[-1]:
            print(f"\n Waiting 10 seconds before next department...")
            time.sleep(10)
    
    # Save combined file if multiple departments
    if len(all_data) > 1:
        save_to_json(all_data, f"{args.output_dir}/all_faculty_combined.json")
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE!")
    print(f"{'='*60}")
    print(f"Output saved to: {args.output_dir}/")
    print(f"Total departments: {len(all_data)}")
    total_faculty = sum(d['faculty_count'] for d in all_data.values())
    print(f"Total faculty: {total_faculty}")


if __name__ == "__main__":
    main()

    # python ku_faculty_scraper.py -d civil_environmental_architectural_engineering
    # https://ceae.ku.edu/faculty