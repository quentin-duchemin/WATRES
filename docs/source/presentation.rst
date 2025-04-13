Presentation of the project
---------------------------



Challenge
~~~~~~~~~

Estimating transit time distributions (TTDs) in watersheds is a critical task for understanding hydrological processes and their impact on water quality, ecosystem health, and resource management. Transit time, defined as the time taken for water parcels to travel from their point of infiltration to a watershed's outlet, provides insight into the movement, storage, and transformation of water and solutes within a watershed system. The characterization of TTDs is essential for deciphering the interaction between surface and subsurface flows, assessing pollutant transport dynamics, and predicting the watershed's response to climatic and anthropogenic changes.

Watersheds are complex systems influenced by a multitude of factors, including topography, soil properties, vegetation, and climatic conditions. These factors introduce variability in flow paths, storage zones, and velocities, which collectively shape the distribution of transit times. As a result, TTDs are inherently heterogeneous and challenging to quantify. Advances in analytical techniques, tracer studies, and modeling approaches have enhanced our ability to estimate TTDs, yet significant uncertainties remain due to data limitations and the simplifications inherent in models.



Description of the method
~~~~~~~~~~~~~~~~~~~~~~~~~

We introduce a new knowledge-guided but data-driven methodology to estimate the transit time distributions of watershed. Our model relies on mixture methods and leverage ideas from survival analysis. The features used by our model are information describing the catchment condition induced by past events such as weighted sum of past precipitation and potential evapotranspiration. 

One can easily add additional features to our model. A typical example is the cosine and sine of the time from January 1st to allow the model to estimate a seasonality pattern. Another example is a time variable allowing the model to learn potential change over time of the catchment behaviour.

Applications of the method
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once trained on a given site for which at least precipitation and streamflow time series are available, one can predict the transit time distributions at any time point. Therefore, our model paves the way to many application to draw hydroglogical insights on the way different catchments behave. 
