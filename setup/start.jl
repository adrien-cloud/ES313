# activate the environment
using Pkg
Pkg.activate(joinpath(homedir(),"OneDrive - Belgian Defence\\Documents\\3BA\\Semester 1\\ES313 - Mathematical Modeling and Computer Simulation\\ES313"))

# change to the root directory of the course
cd(joinpath(homedir(),"OneDrive - Belgian Defence\\Documents\\3BA\\Semester 1\\ES313 - Mathematical Modeling and Computer Simulation\\ES313"))

# start Pluto
using Pluto
Pluto.run()