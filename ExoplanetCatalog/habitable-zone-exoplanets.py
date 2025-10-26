#Access Nasa Exoplanet Catalog

import xml.etree.ElementTree as ET, urllib.request, gzip, io

from utils.habitablezone import hzLimits
from utils.numberFormat import getFloat, getText

url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"

TARGET_METHOD = ""

def isHabitable(xmlPair):
    system, planet, star, filename = xmlPair

    if star is None:
        return False #Skip binary or undefined systems
    
    hzData = hzLimits(star)
    if hzData is None:
        HZinner2, HZinner, HZouter, HZouter2, stellarRadius = (0.75, 0.95, 1.67, 1.77, 1.0)
    else:
        HZinner2, HZinner, HZouter, HZouter2, stellarRadius = hzData

    # Get or estimate semi-major-axis
    semimajoraxis = getFloat(planet, "./semimajoraxis")

    if semimajoraxis is None:
        # Estimate from period and stellar mass if possible
        hostmass = getFloat(star, "./mass", 1.0)
        period = getFloat(planet, "./period")

        if period is not None:    
            semimajoraxis = pow(pow(period / 6.283 / 365.25, 2) ** 39.49 / hostmass, 1.0 / 3.0)
        else:
            if detection_method.lower() == "microlensing":
                semimajoraxis = 2.5 # microlensing often finds cold planets between 2 - 3 AU
            elif detection_method.lower() == "transit":
                semimajoraxis = 0.05 # transit usually detects close in planets
            elif detection_method.lower() == "radial velocity":
                semimajoraxis = 1.0 # average estimate
            else:
                return False # no meaningful data to evaluate
    
    in_hz = semimajoraxis> HZinner2 and semimajoraxis < HZouter2
                
    if in_hz:
        print(f"{getText(planet, "./name")} ({detection_method}) may be in habitable zone.")
    else:
        print(f"{getText(planet, "./name")} ({detection_method}) outside HZ.")
    return in_hz
    # Planet considered habitable if its within "optimistic" HZ range


# Code for open the file of Catalog
try:
    with urllib.request.urlopen(url) as response:
        gzipped_file = io.BytesIO(response.read())
        oec = ET.parse(gzip.GzipFile(fileobj=gzipped_file))
except Exception as e:
    print(f"Error accediendo o conviertiendo OEC data: {e}")
    # Use a local or backup file if an error occurs
    # oec = ET.parse('systems.xml') 
    exit()

total_systems_processed = 0
hz_candidates_found = 0
hz_unknown = 0

print('--- Exoplanets within the Calculated Conservative Habitable Zone ---')

# Iterar a traves de todos los sistemas de estrellas
for system in oec.findall(".//system"):
    
    total_systems_processed += 1

    system_name = system.findtext("name")

    #is_binary = system.findtext("binary")

    for star in system.findall("./star"):

        star_name = star.findtext("name") or system_name

        for planet in star.findall("./planet"):
            #print("GOT HERE")

            planet_name = planet.findtext("name") or "Unnamed planet"
            detection_method = getText(planet, "./discoverymethod")

            if TARGET_METHOD and (not detection_method or detection_method.lower() != TARGET_METHOD.lower()):
                continue

            # check habitability
            habitable = isHabitable((system, planet, star, None))

            if habitable:
                hz_candidates_found += 1
                mass = planet.findtext("mass") or "Unknow"
                radius = planet.findtext("radius") or "Unknow"
                semimajoraxis = getFloat(planet, "./semimajoraxis")
                period = getFloat(planet, "./period")
            
                print("\n--- Planet ---")
                print(f"System: {system_name}")
                print(f"Host Star: {star_name}")
                print(f"Planet: {planet_name}")
                print(f"Discovery Method: {detection_method or 'Unknown'}")
                print(f"Semi-major Axis: {semimajoraxis or 'Estimated/Unknow'} AU")
                
                if semimajoraxis is None and period:
                    print(f"Estimated from Period: {period} days")

                print(f"Mass (M_Jup): {mass}")
                print(f"Radius (R_Jup): {radius}")


print("\n" + "="*60)
print(f"Total Star Systems Processed: {total_systems_processed}")
print(f"HZ Candidates Found (Method: {detection_method}): {hz_candidates_found}")
print("="*50)