#Access Nasa Exoplanet Catalog

import xml.etree.ElementTree as ET, urllib.request, gzip, io

from habitablezone import hzLimits
from ExoplanetCatalog.utils.numberFormat import getFloat, getText

url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"

try:
    with urllib.request.urlopen(url) as response:
        gzipped_file = io.BytesIO(response.read())
        oec = ET.parse(gzip.GzipFile(fileobj=gzipped_file))
except Exception as e:
    print(f"Error accediendo o conviertiendo OEC data: {e}")
    # Use a local or backup file if an error occurs
    # oec = ET.parse('systems.xml') 
    exit()

print('--- Exoplanets within the Calculated Conservative Habitable Zone ---')

# Iterar a traves de todos los sistemas de estrellas
for system in oec.findall("./system"):
    system_name = system.findtext("name")

    host_star = system.find("star")

    # Calculate the habitable zone of the host star
    hz_limits = hzLimits(host_star)

    if hz_limits is None:
        # Cannot calculate HZ (star is missing required data), skip this system
        continue
    

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
        print("No se obtuvo ningun planeta en la zona habitable")


""" oec= ET.parse(gzip.GzipFile(fileobj=io.BytesIO(urllib.request.urlopen(url).read())))

# Mass and radius of all planets
for planet in oec.findall(".//planet"):
    print([planet.findtext("mass"), planet.findtext("radius")]) """