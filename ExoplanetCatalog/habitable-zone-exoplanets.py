#Access Nasa Exoplanet Catalog

import xml.etree.ElementTree as ET, urllib.request, gzip, io

from utils.habitablezone import hzLimits
from utils.numberFormat import getFloat, getText

url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"

TARGET_METHOD = "transit"

def isHabitable(xmlPair):
    system, planet, star, filename = xmlPair

    if star is None:
        return False #Skip binary or undefined systems
    
    hzData = hzLimits(star)
    if hzData is None:
        return False # missing stellar data
    
    HZinner2, HZinner, HZouter, HZouter2, stellarRadius = hzData

    semimajoraxis = getFloat(planet, "./semimajoraxis")
    if semimajoraxis is None:
        # Estimate from period and stellar mass if possible
        hostmass = getFloat(star, "./mass", 1.0)
        period = getFloat(planet, "./period", 265.25)
        semimajoraxis = pow(pow(period / 6.283 / 365.25, 2) ** 39.49 / hostmass, 1.0 / 3.0)

    # Planet considered habitable if its within "optimistic" HZ range
    if semimajoraxis > HZinner2 and semimajoraxis < HZouter2:
        return True
    return False

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

    for host_star_element in system.findall("./star"):

        host_star_name = host_star_element.findtext("name") or system_name

        # Calculate the habitable zone of the host star
        hz_limits = hzLimits(host_star_element)

        if hz_limits is None:
            print(f"Skipping system {system_name} - incomplete stellar data.")
            # Cannot calculate HZ (star is missing required data), skip this system
            continue

        # Use Conservative Habitable Zone Limits 
        HZinner_conservative = hz_limits[1]
        HZouter_conservative = hz_limits[2]

        print(f"System: {system_name}, Host Star: {host_star_name}")

        print(f"HZ inner:  {HZinner_conservative}, HZ Outer: {HZouter_conservative}")

        # Check all planets within specific system

        for planet in host_star_element.findall("./planet"):
            #print("GOT HERE")

            planet_name = planet.findtext("name") or "Unnamed planet"
            detection_method = getText(planet, "./discoverymethod")

            semi_major_axis = getFloat(planet, "./semimajoraxis")
            period = getFloat(planet, "./period")

            if semi_major_axis is None and period is not None:
                hostmass = getFloat(host_star_element, "./mass", 1.0)
                semi_major_axis = ((period / 6.283 / 365.25) ** 2 * 39.49 / hostmass) ** (1.0 / 3.0)


            if semi_major_axis is None:
                #in_hz = None
                # Estimate from period and stellar mass if possible
                hostmass = getFloat(host_star_element,"./mass",1.)
                period = getFloat(planet,"./period",265.25)
                semimajoraxis = pow(pow(period/6.283/365.25,2)*39.49/hostmass,1.0/3.0)
            else: 
                in_hz = HZinner_conservative < semi_major_axis < HZouter_conservative
            
        
            # Does discovery method match the filter
            method_matches = (
                not TARGET_METHOD or
                (detection_method and detection_method.strip().lower() == TARGET_METHOD.lower()
            ))

            if not method_matches:
                continue

            mass = planet.findtext("mass") or "Unknown"
            radius = planet.findtext("radius") or "Unknown"

            if in_hz:
                hz_candidates_found += 1
                hz_status = "In habitable zone"
            elif in_hz is None:
                hz_unknown += 1
                hz_status = "Unknown"
            else:
                continue
            
            print("\n--- Planet ---")
            print(f"System: {system_name}")
            print(f"Host Star: {host_star_name}")
            print(f"Planet: {planet_name}")
            print(f"Discovery Method: {detection_method or 'Unknown'}")
            print(f"Semi-major Axis: {semi_major_axis if semi_major_axis else 'Unknown'} AU")
            print(f"HZ Range: {HZinner_conservative:.3f} – {HZouter_conservative:.3f} AU")
            print(f"Mass (M_Jup): {mass}")
            print(f"Radius (R_Jup): {radius}")
            print(f"HZ Status: {hz_status}")


print("\n" + "="*60)
print(f"Total Star Systems Processed: {total_systems_processed}")
print(f"HZ Candidates Found (Method: {TARGET_METHOD}): {hz_candidates_found}")
print("="*50)

""" 
print('---Habitable Zone Planets---')

for planet in oec.findall(".//planet"):
    if planet.find("habitable") is not None:
        name = planet.findtext("name")
        mass = planet.findtext("mass")
        radius = planet.findtext("radius")

        # Get name of host star/system
        system_name = planet.getParent().findtext("name")

        # Note: 'mass' is in Jupiter masses, and 'radius' is in Jupiter radii
        # The OEC uses a Jupiter-centric unit system for planets.
        
        print(f"System: {system_name}, Planet: {name}, Mass (M_Jup): {mass}, Radius (R_Jup): {radius}")
    else:
        print("No se obtuvo ningun planeta en la zona habitable") """


""" oec= ET.parse(gzip.GzipFile(fileobj=io.BytesIO(urllib.request.urlopen(url).read())))

# Mass and radius of all planets
for planet in oec.findall(".//planet"):
    print([planet.findtext("mass"), planet.findtext("radius")]) """