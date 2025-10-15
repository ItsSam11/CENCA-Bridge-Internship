#Access Nasa Exoplanet Catalog

import xml.etree.ElementTree as ET, urllib.request, gzip, io

from utils.habitablezone import hzLimits
from utils.numberFormat import getFloat, getText

url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"

TARGET_METHOD = "microlensing"

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

print('--- Exoplanets within the Calculated Conservative Habitable Zone ---')

# Iterar a traves de todos los sistemas de estrellas
for system in oec.findall(".//system"):
    
    total_systems_processed += 1

    system_name = system.findtext("name")

    is_binary = system.findtext("binary")

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
            print("GOT HERE")

            planet_name = planet.findtext("name")

            detection_method = getText(planet, "./discoverymethod")

            semi_major_axis = getFloat(planet, "./semimajoraxis")

            if semi_major_axis is None:
                hostmass = getFloat(host_star_name, "./mass", 1.0)
                period = getFloat(planet, "./period", 265.25)
                semimajoraxis = ((period / 6.283 / 365.25) ** 2 * 39.49 / hostmass) ** (1.0 / 3.0)

            # Is in habitable zone?
            is_in_hz = (
                semi_major_axis is not None and
                HZinner_conservative < semi_major_axis < HZouter_conservative
            )
        

            # Does discovery method match the filter
            method_matches = (
                detection_method and
                detection_method.strip().lower() == TARGET_METHOD.lower()
            )

            if is_in_hz:

                hz_candidates_found += 1

                mass = planet.findtext("mass")
                radius = planet.findtext("radius")

                print("--- Habitable Zone Candidate Found ---")
                print(f"System : {system_name}, Planet : {planet_name}")
                print(f"Discovery Method : {detection_method}")
                print(f"Orbit : {semi_major_axis:.3f}AU")
                print(f"Conservative HZ Range : {HZinner_conservative:.3f} to {HZouter_conservative:.3f}")
                print(f"Planet Properties (M_Jup, R_Jup): {mass}, {radius}")


print("\n" + "="*50)
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