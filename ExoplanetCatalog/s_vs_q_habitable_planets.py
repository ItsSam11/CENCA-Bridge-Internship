# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from oec_filters import isHabitable, isFiltered
import oec_fields
import xml.dom.minidom
from bs4 import BeautifulSoup

import xml.etree.ElementTree as ET, urllib.request, gzip, io
url = "https://github.com/OpenExoplanetCatalogue/oec_gzip/raw/master/systems.xml.gz"
oec = ET.parse(gzip.GzipFile(fileobj=io.BytesIO(urllib.request.urlopen(url).read())))

def extract_numeric(html_str):
    if pd.isna(html_str):
        return None
    # Parse the HTML to extract text (the numeric value)
    soup = BeautifulSoup(html_str, 'html.parser')
    text = soup.get_text()
    # Remove uncertainty part by splitting on 
    numeric_part = text.split('±')[0].strip()
    try:
        return float(numeric_part)
    except ValueError:
        return None

# Assuming 'tree' is your parsed ElementTree object and 'root' is root element
root = oec.getroot()

# Example: You have filename as a string from your data source
filename_str = "systems.xml"  # Replace with your actual filename variable

# Create a new XML Element named 'filename' and set its text content to the filename string
filename_element = ET.Element('filename')
filename_element.text = filename_str


xmlPairs = []

for system in root.findall('system'):

    

    stars = system.findall('.//star')

    for planet in system.findall('.//planet'):
        

        # Choose star element closest to this planet (example: first star)
        star = stars[0] if stars else None
        

        pair = {
            'system': system,  # XML Element for system name only
            'planet': planet,  # XML Element for planet name only
            'star': star,      # XML Element or None
            'filename': filename_element         # string filename
        }

        xmlPairs.append(pair)
p = []
fields = ["massEarth", "semimajoraxis", "starmass"]
lastfilename = ""
tablecolour = 0
for xmlPair in xmlPairs:

    if isFiltered(xmlPair,['habitable']):
        continue
    
    # Extract values from the dict by keys
    system = xmlPair['system']
    planet = xmlPair['planet']
    star = xmlPair['star']
    filename = xmlPair['filename']
    



    # Rest of your code ...
    if lastfilename != filename.text:  # filename is an Element; get text
        lastfilename = filename.text
        tablecolour = not tablecolour
  
    d = {}
    d["fields"] = [tablecolour]

    for field in fields:

        d["fields"].append(oec_fields.render(xmlPair, field, editbutton=False))
    p.append(d)

# Extract rows as lists
rows = [pl['fields'] for pl in p]

# Define column names (make sure columns count matches fields count)
columns = ['Selected', 'Mass', 'SemiMajorAxis', 'StarMass']

# Create DataFrame
df = pd.DataFrame(rows, columns=columns)

# Assuming df is your DataFrame and columns like 'Mass' contain HTML strings
for col in ['Mass', 'SemiMajorAxis', 'StarMass']:  # add other columns as needed
    df[col] = df[col].apply(extract_numeric)

df = df.sort_values(by='SemiMajorAxis', ascending=True).dropna()

print(len(df))


# Load the CSV data
file_path = 'ML_2025.10.13_08.13.58.csv'
data = pd.read_csv(file_path, comment='#')

# Convert columns to numeric
data['sy_dist'] = pd.to_numeric(data['sy_dist'], errors='coerce')
data['ml_dists'] = pd.to_numeric(data['ml_dists'], errors='coerce')

# Calculate (D_LS * D_L / D_S), where D_LS = D_S - D_L
D_L = data['sy_dist']
D_S = data['ml_dists']
D_LS = D_S - D_L
ratio = (D_LS * D_L) / D_S

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(ratio.dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title(r'Histogram of $(D_{LS}D_L / D_S)$')
plt.xlabel(r'$(D_{LS}D_L / D_S)$ [pc]')
plt.ylabel('Number of systems')
plt.grid(True)
plt.show()

# Calculate and print percentiles
percentiles = np.percentile(ratio.dropna(), [26, 50, 84])
print("26th percentile:", percentiles[0])
print("50th percentile (median):", percentiles[1])
print("84th percentile:", percentiles[2])

G = 6.6743e-11
c = 2.998e8
M_o_to_kg = 1.9885e30
pc_to_m = 3.086e16
m_to_AU = 6.6846e-12

df['PlanetSolarMass'] = df['Mass'].values*3e-6

df['q'] = df['PlanetSolarMass'].values / df['StarMass'].values

df['RE_Median'] = np.sqrt(((4*G)/c**2)*(df['StarMass'].values*M_o_to_kg)*percentiles[1]*pc_to_m)
df['RE_upper'] = np.sqrt(((4*G)/c**2)*(df['StarMass'].values*M_o_to_kg)*percentiles[2]*pc_to_m)
df['RE_lower'] = np.sqrt(((4*G)/c**2)*(df['StarMass'].values*M_o_to_kg)*percentiles[0]*pc_to_m)

df['s_Median'] = df['SemiMajorAxis'].values / (df['RE_Median'].values*m_to_AU)
df['s_upper'] = df['SemiMajorAxis'].values / (df['RE_upper'].values*m_to_AU)
df['s_lower'] = df['SemiMajorAxis'].values / (df['RE_lower'].values*m_to_AU)

s_err_up = df['s_lower'].values - df['s_Median'].values
s_err_down = df['s_Median'].values - df['s_upper'].values

errors = [s_err_up, s_err_down]

plt.semilogx([],[])
plt.scatter(df['q'].values, df['s_Median'].values)
plt.errorbar(df['q'].values, df['s_Median'].values, yerr = errors, linestyle = 'none')
plt.xlabel('Mass Ratio, q')
plt.ylabel('s')
plt.show()



from scipy.stats import gaussian_kde

x = df['q'].values
y = df['s_Median'].values

# Evaluate a gaussian kde on a grid
xy = np.vstack([x, y])
kde = gaussian_kde(xy)
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

# Create grid points
X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
positions = np.vstack([X.ravel(), Y.ravel()])

# Evaluate densities on the grid
Z = kde(positions).reshape(X.shape)

# Plot contour and scatter points
plt.semilogx()  # maintain log scale in x as original
plt.scatter(x, y, s=10, c='black', alpha=0.3, label='Data points')
contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')
plt.colorbar(contour, label='Density')
margin = 0.01
plt.xlim(df['q'].min() - 1e-7, df['q'].max() + 0.05)
plt.ylim(df['s_Median'].min() - 0.05, df['s_Median'].max() + 0.1)
plt.xlabel('Mass Ratio, q')
plt.ylabel('s')
plt.title('Density Contour Plot')
plt.show()























