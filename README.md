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