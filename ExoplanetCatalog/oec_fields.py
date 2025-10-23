import urllib
import lxml.etree as ET
from numberformat import renderFloat, renderText, notAvailableString

""" Array of title of propoerties """
titles = {
    "name":                         "Primary planet name",
    "namelink":                     "Primary planet name",
    "alternativenames":             "Alternative planet names",
    "starname":                     "Star name",
    "staralternativenames":         "Alternative star names",
    "systemname":                   "Primary system name",
    "systemalternativenames":       "Alternative system names",
    "distance":                     "Distance [parsec]",
    "distancelightyears":           "Distance [lightyears]",
    "numberofplanets":              "Number of planets in system",
    "numberofstars":                "Number of stars in system",
    "rightascension":               "Right ascension",
    "declination":                  "Declination",
    "image":                        "Image",
    "imagedescription":             "Image description",
    "starmass":                     "Mass [M<sub>Sun</sub>]",
    "starradius":                   "Radius [R<sub>Sun</sub>]",
    "starage":                      "Age [Gyr]",
    "starmetallicity":              "Metallicity [Fe/H]",
    "starspectraltype":             "Spectral type",
    "startemperature":              "Temperature [K]",
    "starmagV":                     "Visual magnitude",
    "period":                       "Orbital period [days]",
    "semimajoraxis":                "Semi-major axis [AU]",
    "eccentricity":                 "Eccentricity",
    "temperature":                  "Equilibrium temperature [K]",
    "lists":                        "Lists",
    "description":                  "Description",
    "discoveryyear":                "Discovery year",
    "discoverymethod":              "Discovery method",
    "lastupdate":                   "Last updated [yy/mm/dd]",
    "mass":                         "Mass [M<sub>jup</sub>]",
    "radius":                       "Radius [R<sub>jup</sub>]",
    "massEarth":                    "Mass [M<sub>earth</sub>]",
    "radiusEarth":                  "Radius [R<sub>earth</sub>]",
}

def getEditButton(xmlPair,o):
    if o is None:
        return ""
    else:
        system, planet, star, filename = xmlPair
        path = ET.ElementTree(system).getpath(o)
        if path is not None:
            return "<a class='editbutton' href='/edit/form/"+filename+path[7:]+"'>edit</a>"
    return ""

def render(xmlPair,kind,editbutton=True):
    editbutton = False ## Manual override (June 21 2020)
    # Extract values from the dict by keys
    system = xmlPair['system']
    planet = xmlPair['planet']
    star = xmlPair['star']
    filename = xmlPair['filename']
    if kind=="numberofplanets":
        return "%d"%len(system.findall(".//planet"))
    if kind=="numberofstars":
        return "%d"%len(system.findall(".//star"))
    if kind in ["distance"]:
        o = system.find("./"+kind)
        html = renderFloat(o)
        if editbutton:
            html += getEditButton(xmlPair,o)
        return html
    if kind=="distancelightyears":
        return renderFloat(system.find("./distance"),3.2615638)
    if kind=="massEarth":
        return renderFloat(planet.find("./mass"),317.82841)
    if kind in ["mass","radius","period","eccentricity","temperature","semimajoraxis"]:
        o = planet.find("./"+kind)
        print(o)
        html = renderFloat(o)
        if editbutton:
            html += getEditButton(xmlPair,o)
        return html
    if kind=="radiusEarth":
        return renderFloat(planet.find("./radius"),11.208981)
    # Text based object
    if kind=="rightascension":
        return renderText(system.find("./rightascension"))
    if kind=="declination":
        return renderText(system.find("./declination"))
    if kind=="image":
        try:
            return planet.find("./image").text
        except:
            return None
    if kind=="imagedescription":
        try:
            return planet.find("./imagedescription").text
        except:
            return None
    if kind=="description":
        o = planet.find("./description")
        html = renderText(o)
        if editbutton:
            html += getEditButton(xmlPair,o)
        return html
    if kind=="name":
        return renderText(planet.find("./name"))
    if kind=="namelink":
        planetname = planet.find("./name").text
        return "<a href=\"/planet/%s/\"><span class=\"numericvalue\">%s</span></a>"%(urllib.parse.quote(planetname.encode('utf8')),planetname)
    if kind=="discoveryyear":
        return renderText(planet.find("./discoveryyear"))
    if kind=="discoverymethod":
        return renderText(planet.find("./discoverymethod"))
    if kind=="lastupdate":
        return renderText(planet.find("./lastupdate"))
    if kind=="systemname":
        return renderText(system.find("./name"))
    if kind=="alternativenames":
        alternativenames = notAvailableString 
        names = planet.findall("./name")
        for i,name in enumerate(names[1:]):
            if i==0:
                alternativenames = ""
            else:
                alternativenames += ", "
            alternativenames += name.text
        return alternativenames
    if kind=="systemalternativenames":
        systemalternativenames = notAvailableString 
        systemnames = system.findall("./name")
        for i,name in enumerate(systemnames[1:]):
            if i==0:
                systemalternativenames = ""
            else:
                systemalternativenames += ", "
            systemalternativenames += name.text
        return systemalternativenames
    if kind=="lists":
        lists = notAvailableString 
        ls = planet.findall("./list")
        for i,l in enumerate(ls):
            if i==0:
                lists = ""
            else:
                lists += "; "
            lists += l.text
        return lists
    # Host star fields
    if kind[0:4]=="star":
        if star is None:
            return notAvailableString
        kind = kind[4:]
        # Text based object
        if kind=="spectralkind":
            return renderText(star.find("./spectralkind"))
        if kind=="name":
            return renderText(star.find("./name"))
        if kind=="alternativenames":
            alternativenames = notAvailableString 
            names = star.findall("./name")
            for i,name in enumerate(names[1:]):
                if i==0:
                    alternativenames = ""
                else:
                    alternativenames += ", "
                alternativenames += name.text
            return alternativenames
        if kind in ["mass","radius","age","metallicity","temperature","magV"]:
            o = star.find("./"+kind)
            html = renderFloat(o)
            if editbutton:
                html += getEditButton(xmlPair,o)
            return html
        # Default: just search for the property in the planet xml. 
        return renderFloat(star.find("./"+kind))
    # Long texts
    if kind=="systemcategory":
        systemcategory = ""
        systemname = renderText(system.find("./name"))
        if len(system.findall(".//planet"))==1:
            systemcategory += "The planetary system "+systemname+" hosts at least one planet. "
        elif len(system.findall(".//planet"))>1:
            systemcategory += "The planetary system "+systemname+" hosts at least %d planets. " % len(system.findall(".//planet"))
        if len(system.findall(".//star"))>1:
            systemcategory += "Note that the system is a multiple star system. It hosts at least %d stellar components. "% len(system.findall(".//star"))
        elif len(system.findall(".//star"))==0:
            systemcategory += "The planet is a so called orphan planet and not associated with any star. "
        return systemcategory

    # Default: just search for the property in the planet xml. 
    return renderFloat(planet.find("./"+kind))
