# -*- coding: utf-8 -*-
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from oec_filters import isHabitable, isFiltered
import oec_fields
import xml.dom.minidom
from bs4 import BeautifulSoup

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


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

    system_name = system.find('name').text if system.find('name') is not None else 'Unknown System'
    planet_name = planet.find('name').text if planet.find('name') is not None else  'Unknown System'
    d['System'] = system_name
    d['Planet'] = planet_name

    for field in fields:
        d["fields"].append(oec_fields.render(xmlPair, field, editbutton=False))

    d['fields'].append(system_name)
    d['fields'].append(planet_name)

    p.append(d)

# Extract rows as lists
rows = [pl['fields'] for pl in p]

# Define column names (make sure columns count matches fields count)
columns = ['Selected', 'Mass', 'SemiMajorAxis', 'StarMass', 'System', 'Planet']

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

# ------ CONSTANTS -----
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

# Main contour plot indicating the number of systems per unit of data
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

# ---- Find and plot top peaks ----
# Find peak indices values programatically

# --- Smooth the heatmap slightly to highlight real peaks ---
smoothed = gaussian_filter(heatmap, sigma=1.5)

# Get bin centers (for plotting & coordinate conversion)
xcenters = 0.5 * (xedges[:-1] + xedges[1:])
ycenters = 0.5 * (yedges[:-1] + yedges[1:])

# --- Detect density peaks automatically ---
peaks = peak_local_max(
    smoothed, 
    min_distance=1, # smaller, more peaks allowed
    threshold_rel=0.10, #smaller, includer weaker peaks
    num_peaks=4 # top 4 strongest peaks
)

# Convert peak indices to q, s coordinates
q_peaks = xcenters[peaks[:, 0]]
s_peaks = ycenters[peaks[:, 1]]

print("\nDetected density peaks:")
for i, (qv, sv) in enumerate(zip(q_peaks, s_peaks), 1):
    print(f"Peak {i}: q ≈ {qv:.2e}, s ≈ {sv:.3f}")

# ---- Plot contour + peaks ----
plt.figure(figsize=(9,6))
plt.semilogx()
plt.scatter(x, y, s=10, c='gray', alpha=0.4, label='Data points')
contour = plt.contourf(xcenters, ycenters, smoothed.T, levels=25, cmap='viridis')
plt.colorbar(contour, label='Density')

# Mark all peaks with red stars
plt.scatter(q_peaks, s_peaks, color='red', marker='*', s=200, label='Density peaks')

# Optional: label peaks with numbers
for i, (qx, sy) in enumerate(zip(q_peaks, s_peaks), 1):
    plt.text(qx * 1.05, sy, f"{i}", color='red', fontsize=10, fontweight='bold')

# Formatting
plt.xlim(df['q'].min()*0.8, df['q'].max()*1.5)
plt.ylim(0, df['s_Median'].max() + 0.1)
plt.xlabel('Mass Ratio, q')
plt.ylabel('s')
plt.title('Microlensing Density Peaks (Gaussian-smoothed 2D histogram)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# --- Print detected peaks ---
print("\nDetected density peaks:")
for i, (qv, sv) in enumerate(zip(q_peaks, s_peaks), 1):
    print(f"Peak {i}: q ≈ {qv:.2e}, s ≈ {sv:.3f}")

# ==============================================================
# === FIND SYSTEMS CLOSE TO THE MAIN DENSITY PEAK(S) ===========
# ==============================================================

# Parameters: adjust to control how close "near" means
log_q_radius = 0.3      # allowed spread in log10(q)
s_radius = 0.15         # allowed spread in s
d_thresh = 0.6          # combined distance threshold (dimensionless)

# Convert to arrays for convenience
q_values = df['q'].values
s_values = df['s_Median'].values
logq = np.log10(q_values)

for i, (q_center, s_center) in enumerate(zip(q_peaks, s_peaks), 1):
    logq_center = np.log10(q_center)

    # Distance metric combining log(q) and s differences
    d_vals = np.sqrt(((logq - logq_center) / log_q_radius)**2 + ((s_values - s_center) / s_radius)**2)
    
    mask = d_vals <= d_thresh

    nearby = df[mask].copy()

    print(f"\n===== Peak {i}: q ≈ {q_center:.2e}, s ≈ {s_center:.3f} =====")
    print(f"Search window: ±{log_q_radius:.2f} dex in q, ±{s_radius:.2f} in s")
    print(f"Found {len(nearby)} systems near this density peak.\n")

    if len(nearby) > 0:
        # Display key columns
        cols_to_show = ['System', 'Planet' ,'Mass', 'SemiMajorAxis', 'StarMass', 'q', 's_Median']
        print(nearby[cols_to_show].sort_values(by='q').to_string(index=False))
    else:
        print("No systems found near this peak. Try relaxing thresholds.")

# ----- Calculate Einteins Radius and Einsteing Time for 4 systems -----

# Note: Planet mass is in Earth masses (M_earth), Star mass is in Solar masses (M_sun)
systems = pd.DataFrame({
    'System': ['HD 210277 b', 'HD 128356 b', 'kappa CrB b', 'HD 141399 d'],
    'PlanetMass_Earth': [391.0, 283.0, 509.0, 375.0],
    'StarMass_Solar': [1.09, 0.65, 1.51, 1.07]
})

# ------ CONSTANTS -----
G = 6.6743e-11 # Gravitational Constant (m^3 kg^-1 s^-2)
c = 2.998e8 # Speed of light (m/s)
M_sun_to_kg = 1.989e30 # Solar Mass (kg)
M_earth_to_kg = 5.972e24 # Earth Mass (kg)  
pc_to_m = 3.086e16 # Parsec to meter (m/pc)  
muas_to_rad = 4.8481e-12 # micro-arcsecond to radian (rad/muas)

# --- Microlensing Geometry Parameters ---
D_l_pc = 6770.0 # Lens Distance (pc)
tilde_D_pc = 1502.0 # Effective Distance Term (D_L * D_LS / D_S) (pc)
mu_LS_muas_per_day = 15.0 # Relative proper motion (muas/day)

# --- 1. Total Lens Mass (M_L) in kg ---
# Convert masses to kg and sum them to get the total lens mass
systems['PlanetMass_kg'] = systems['PlanetMass_Earth'] * M_earth_to_kg
systems['StarMass_kg'] = systems['StarMass_Solar'] * M_sun_to_kg
systems['TotalMass_kg'] = systems['PlanetMass_kg'] + systems['StarMass_kg']

# --- 2. Effective Distance Term (tilde_D) in m ---
tilde_D_m = tilde_D_pc * pc_to_m

# --- 3. Angular Einstein Radius ($\theta_E$) ---
# Formula for R_E (in m): R_E = sqrt( (4G * M_L / c^2) * tilde_D )
systems['R_E_m'] = np.sqrt((4 * G * systems['TotalMass_kg'] / c**2) * tilde_D_m)

# Convert R_E to angular Einsteins Radius ($\theta_E$) in radians: theta_E = R_E / D_L
systems['theta_E_rad'] = systems['R_E_m'] / (D_l_pc * pc_to_m) 

# Convert theta from radians to micro-arcseconds
systems['EinsteinRadius_muas'] = systems['theta_E_rad'] / muas_to_rad

# --- 4. Einstein Time ($t_E$) in days ---
# Formula for t_E (in days): t_E = $\theta_E$ / $\mu_{LS}$
systems['EinsteinTime_days'] = systems['EinsteinRadius_muas'] / mu_LS_muas_per_day

# Final Output 
final_results = systems[['System', 'EinsteinRadius_muas', 'EinsteinTime_days']].copy()

print("--- Microlensing Parameters for 4 selected systems ---")
print(f"Effective Distance Term (D_L*D_LS/D_S): {tilde_D_pc} pc")
print(f"Relative Proper Motion (mu_LS): {mu_LS_muas_per_day} muas/day\n")

print("--- Calculated Einstein Radius (theta_E) and Einstein Time (t_E) ---")
print("| System | Einstein Radius ($\mathbf{\mu\text{as}}$) | Einstein Time ($\mathbf{days}$) |")
print(final_results.round(2).to_markdown(index=False, numalign="center"))

#print(systems.dtypes)
#print(systems[['System', 'EinsteinRadius_muas', 'EinsteinTime_days']])

df.to_csv('Habitable_Planet_microlensing.csv', index = False)






















