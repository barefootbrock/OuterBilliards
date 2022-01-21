from outerbilliards import PolygonBilliards
from geometry import LineSet
import utils

#Settings
nSides = 5 #Number of sides
iterations = 100 #Number of iterations
length = 25 #Initial length of line segments
edgeMethod = PolygonBilliards.REFLECT_NONE #What to do with points/line segments that land on singularity
useSymmetry = True #Use symmetry to speed up calculation


#Create regular polygon outer billiard object
B = PolygonBilliards.regularPolygon(nSides, singularityLen=length)

 #Get line segments for the billiard's singularity (this is iteration 0)
singularity = B.singularity()
if useSymmetry:
    singularity = LineSet(singularity[0,...])

prevLines = [] #Empty array for previous lines

#Loop through each iteration
for i in range(iterations):
    prevLines.append(singularity) #Keep a reccord of the previous line segments

    singularity = B(singularity) #Apply the billiard transformation
    singularity = singularity.simplify() #Join overlapping line segments

    #Set edge method after first iteration
    if i == 0:
        B.setEdgeMethod(edgeMethod)

    print("Iteration", i)


#Join the results from each iteration together
result = singularity.union(*prevLines, simplfy=True)
if useSymmetry:
    result = utils.applySymmetry(result, rotational=nSides)

B.plot(color="black") #Plot the inner polygon
result.plot() #Plot the resulting line segments

utils.showPlot() #Actually show the plot