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



# Calculate and print percentiles
percentiles = np.percentile(ratio.dropna(), [26, 50, 84])
print("26th percentile:", percentiles[0])
print("50th percentile (median):", percentiles[1])
print("84th percentile:", percentiles[2])

# Find indices corresponding to values closest to median ratio
median_value = percentiles[1]

# To work safely with NaNs, get a mask of valid indices
valid_mask = ratio.notna()
valid_indices = np.where(valid_mask)[0]

# Extract valid ratio values
valid_ratio = ratio[valid_mask]

# Find index with closest ratio to median
closest_idx = valid_indices[np.argmin(np.abs(valid_ratio - median_value))]

# Extract corresponding D_L and D_S values
median_D_L = D_L.iloc[closest_idx] if hasattr(D_L, 'iloc') else D_L[closest_idx]
median_D_S = D_S.iloc[closest_idx] if hasattr(D_S, 'iloc') else D_S[closest_idx]
median_ratio = ratio[closest_idx]

print(f'Values at median ratio (~{median_value:.4f}):')
print('Ratio = ', ratio[closest_idx])
print('D_L =', median_D_L)
print('D_S =', median_D_S)

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(ratio.dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title(r'Histogram of $(D_{LS}D_L / D_S)$')
plt.ylim(0, 21)
plt.plot([percentiles[0], percentiles[0]], [0,25], color = 'black', linestyle = '--')
plt.plot([percentiles[2], percentiles[2]], [0,25], color = 'black', linestyle = '--')
plt.plot([percentiles[1], percentiles[1]], [0,25], color = 'black', label = 'Median')
plt.plot([median_ratio, median_ratio], [0,25], color = 'red', label = 'Selected System')
plt.xlabel(r'$(D_{LS}D_L / D_S)$ [pc]')
plt.ylabel('Number of systems')
plt.grid(True)
plt.legend()
plt.show()

G = 6.6743e-11
c = 2.998e8
M_o_to_kg = 1.9885e30
pc_to_m = 3.086e16
m_to_AU = 6.6846e-12

df['PlanetSolarMass'] = df['Mass'].values*3e-6

df['q'] = df['PlanetSolarMass'].values / df['StarMass'].values

df['RE_Median'] = np.sqrt(((4*G)/c**2)*(df['StarMass'].values*M_o_to_kg)*median_ratio*pc_to_m)
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



x = df['q'].values
y = df['s_Median'].values

# Define bins in x and y, optionally add margins as per needs
bins_x = np.logspace(np.log10(x.min()*0.5), np.log10(x.max()*1.5), 30)
bins_y = np.linspace(y.min()*0.5, y.max()*1.5, 30)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y])

# Plot contours of the binned counts
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

plt.semilogx()  # maintain log x scale
plt.scatter(x, y, s=10, c='black', alpha=0.3, label='Data points')
contour = plt.contour(X, Y, heatmap.T, levels=10, cmap='viridis')
plt.colorbar(contour, label='Count Density')
plt.xlim(df['q'].min() - 1e-7, df['q'].max()*1.5)
plt.ylim(0, df['s_Median'].max() + 0.1)
plt.xlabel('Mass Ratio, q')
plt.ylabel('s')
plt.title('Density Contour via 2D Histogram')
plt.show()


df.to_csv('Habitable_Planet_microlensing.csv', index = False)





















