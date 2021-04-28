# Volume/area scaling model

This repository contains all code files used to run experiments, process the results and plot figures for the VAS paper.

The paper is the continuation of my Master's thesis "Testing the importance of explicit ice dynamic for mountain glacier change projections". For this work I re-implemented the volume/area scaling model by [Marzeion et al. (2012)][Marzeion et al. (2012)] into the Open Global Glacier Model ([OGGM](https://oggm.org/)) framework. This allowed to perform a set of different experiments comparing the OGGM flowline model to the scaling model under controlled boundary conditions.

The model implementation can be found under [github.com/OGGM/oggm-vas](https://github.com/OGGM/oggm-vas).

## Content

- `code`: Contains all the code (mainly python scripts) used to run the model and/or plot results. The code for the volume/area scaling model itself has a dedicated OGGM repository.



[Marzeion et al. (2012)]: https://doi.org/10.5194/tc-6-1295-2012	"Marzeion, B., Jarosch, A. H., and Hofer, M.: Past and future sea-level change from the surface mass balance of glaciers, The Cryosphere, 6, 1295–1322, https://doi.org/10.5194/tc-6-1295-2012, 2012."

