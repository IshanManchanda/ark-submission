# ARK Task 3: Pathfinding & CV Puzzle

### Project Structure
The Jupyter notebook solution.ipynb contains the solutions
to levels 1, 2, and 4. The maze generated for level 3 is visualized in
the same notebook, but the level has been solved using a separate script.

level3.py is the entry-point into the solution for this level,
which imports the PathFinder class from pathfinder.py
and the various Heuristic classes from heuristics.py.

All given images and inputs are present in the data directory while the 
generated images and the results are stored in the out directory.

All the Heuristic classes implement the Heuristic interface
which is implemented in Python as an Abstract Base Class.
