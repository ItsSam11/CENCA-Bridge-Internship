import MulensModel
import matplotlib.pyplot as plt
import os.path

# Define a point lense model - PSPL (Point Source, Point Lens)
my_pspl_model = MulensModel.Model(
    {
        't_0': 2452848.06, #time of closets approach BETWEEN SOURCE AND LENS
        'u_0': 0.133,   # impact parameter: it defines the magnification impact at the of the minimum angle closest approach. 
                        # its measured in units of einstein radious.
                        # TELLS ME HOW CLOSE THE SOURCE STARS PATH COMES TO THE LENS.
                        #value by the closets angle separation between the source and lens
        't_E': 61.5 # einstein crossing time: time it takes to the source to cross a distance equals to einsteins ring of the lens.
    }
)

# Model with two bodies - 1S2L (1 Source, 2 Lens)
my_1S2L_model = MulensModel.Model(
    {
        't_0': 2452848.06,
        'u_0': 0.133, 
        't_E': 61.5,
        'rho': 0.00096,
        'q': 0.0039,
        's': 1.120,
        'alpha': 223.8
    }
)

# since rho is set, define a time range and method to apply finite source effects --- EXPLAIN THIS A BIT BETTER
my_1S2L_model.set_magnification_methods(
    [2452833., 'VBBL', 2452845.]
)

# Plot the models
my_pspl_model.plot_magnification(
    t_range=[2452810, 2452890], subtract_2450000=True, color="red",
    linestyle=":", label="PSPL"
)   

my_1S2L_model.plot_magnification(
    t_range=[2452810, 2452890], subtract_2450000=True, color="black",
    label="1S2L"
)

""" plt.ylim(0,14)
plt.legend(loc='best')
plt.show() """

# Imported Data

# Optical Gravitational Lensing Experiment
OGLE_data = MulensModel.MulensData(
    file_name=os.path.join(
        MulensModel.DATA_PATH, "photometry_files", "OB03235", "OB03235_OGLE.tbl.txt"
    ),
    comments=['\\', '|'], plot_properties={'label': 'OGLE', 'color': 'orange'}
) 

# Microlensing Observations in Astrophysics
MOA_data = MulensModel.MulensData(
    file_name=os.path.join(
        MulensModel.DATA_PATH, "photometry_files", "OB03235", "OB03235_MOA.tbl.txt"
    ),
    phot_fmt='flux', comments=['\\', '|'], plot_properties={'label': 'MOA', 'color': 'cyan'}
)

# Combine the two together
my_event = MulensModel.Event(
    datasets=[MOA_data, OGLE_data], model=my_1S2L_model
)

# Plot the result

my_event.plot_model(
    t_range=[2452810, 2452890], subtract_2450000=True, color='black',
    data_ref=1
)

my_event.plot_data(
    subtract_2450000=True, data_ref=1, s=5, markeredgecolor='gray'
)

# Mulens automatically fits to the source and blend flux for the given model

# Customize the output
plt.legend(loc='best')
plt.title('OGLE-2003-BGL-235/MOA-2003-BGL-53')
plt.ylim(19., 16.5)
plt.xlim(2810, 2890)
plt.show()