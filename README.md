# ComputationalScienceProject
This is the repository for the project Computational Science course. Built by team 15 that focusses on rocket trajectory.

# requirements
- numpy
- matplotlib
- pandas
- opencv-python
- scipy

# Model Assumptions/Limitations
- The rocket is treated as a 2D system:
  - No lift, pitch, roll, yaw considered in aerodynamics
  - No rotation due to off-centered thrust or side forces
  - Drag resistance scales with area and velocity but forces are only considered as a point at the center of mass
- Wind shear and turbulence were not considered
- Thrust and propellant flow rate are constant
- The rocket is launched at sea level
- The earth has no local differences in gravity constants
- There is instantaneous drop off of rocket mass during staged detachments

# Usage
- Download the repository to your system.
- Download the dependencies using pip3 install -r requirements.txt
- Run the main.py obtain data function / plot function for the angle.
  - 1:
        plot_while_processing()
  - 3:
        obtain_thrust_data_from_simulation()
  - 4:
        obtain_data_from_launchangle_simulation(startlaunch, endlaunch, N)
  - 5:
        obtain_optimal_angle()
  - 6:
        obtain_poster_launchangle_pic()
  - 7:
        obtain_zoomed_in_launchangle_pic()
  - else:
        plot_thrust_data_from_simulation()

# Reproduction tips
- ModelData isn't included in the folder, because the file sizes are too big,
  thus to reproduce the obtaining code of the best angle we recommend running
  the following set of commands all in seperate terminals:
      - ```python main.py 4 0 15 16```
      - ```python main.py 4 15 30 16```
      - ```python main.py 4 30 45 16```
      - ```python main.py 4 45 60 16```
      - ```python main.py 4 60 75 16```
      - ```python main.py 4 75 90 16```
  Then to actually obtain the optimal angle simply run:
  ```python main.py 5```