# Algorithm Modes

## MOGA
Baseline binary-coded MOGA with roulette-style selection, uniform crossover, bit mutation, and dominance-based ranking.

## MOGA Bonus
Adds tournament selection, elitism, adaptive mutation, and an external nondominated archive.

## Crowding
Adds NSGA-II-style crowding distance to parent selection, elitism, and archive pruning.

## Crowding + epsilon
Adds epsilon-grid archive filtering to reduce near-duplicate archive points and control objective-space spread.

## Crowding + HV
Adds hypervolume-contribution archive pruning. Boundary points are preserved, and low exclusive-contribution points are removed first when the archive exceeds the allowed size.
