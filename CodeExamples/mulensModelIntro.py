import MulensModel
import matplotlib.pyplot as plt
import os.path

# Define a point lense model
my_pspl_model = MulensModel.Model(
    {
        't_0': 2452848.06, #time of closets approach
        'u_0': 0.133, # impact parameter: defines the magnification impact value by the closets angle separation between the source and lens
        't_E': 61.5 # einstein crossing time: time it takes to the source to cross a distance equals to einsteins ring of the lens.
    }
)

# Model with two bodies
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

plt.legend(loc='best')
plt.show()