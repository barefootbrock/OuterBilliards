# OuterBilliards
This code is associated with the presentation *Outer Billiard Visualization Algorithms*, which I (David Brock) am working on with Byungik Kahng from the University of North Texas at Dallas (UNTD). Its main purpose is to generate and display the singularity structure of outer/dual billiards on polygons, especially regular polygons. 

Below is our abstract:

>We study the singularity structure of outer billiards (or dual billiards) on regular polygons, which often gives rise to intricate and beautiful fractal structures. Although this is not a new topic, little (if any) research exists on the efficiency of calculation and visualization of this structure. First, we present a baseline algorithm, primarily for comparison. Then, by leveraging key properties of the outer billiards transformation, several major improvements are made to this algorithm. Compared to the baseline, the improved algorithms run faster, use less memory, and generate higher resolution images. In addition to being beautiful in its own right, this research should help others in the field of polygonal outer billiards by allowing for quick visualization and experimentation.

The required Python libraries are Numpy, Matplotlib, and Scipy (only for smooth inner/outer billiards). Once you have these, demo.py is a good place to start.

If you find this useful, please cite it.
