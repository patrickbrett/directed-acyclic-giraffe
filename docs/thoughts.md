Imagine a map where the characters are split off from one another and/or one is enclosed, it may be a very good idea to rent the portal gun / bike to get from a to b.

modified version of A* somehow taking into account distances to valued items could be a good metric... would be an optimisation compared to naive A* for each square which might run too slow

---------

some ways to incorporate A*:

- to start with, just A* to the point 0,0 to demonstrate the algo works
- then, incorporate A*-ing to the cell with the most points on the board
- then, incorporate calculating A* distance to the point with highest Q score (I'm defining Q score as points / manhattan distance)
- can also cache the A* result - if we're on the path from A to B and are still aiming for B, then the result is still valid
- what about finding other things along the way?

--------------

map 6 is an interesting one - it has clear "worst squares" in terms of distance to reach things
