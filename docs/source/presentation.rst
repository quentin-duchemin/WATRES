Presentation of the project
---------------------------



Challenge
~~~~~~~~~

Estimating transit time distributions (TTDs) in watersheds is a critical task for understanding hydrological processes and their impact on water quality, ecosystem health, and resource management. Transit time, defined as the time taken for water parcels to travel from their point of infiltration to a watershed's outlet, provides insight into the movement, storage, and transformation of water and solutes within a watershed system. The characterization of TTDs is essential for deciphering the interaction between surface and subsurface flows, assessing pollutant transport dynamics, and predicting the watershed's response to climatic and anthropogenic changes.

Watersheds are complex systems influenced by a multitude of factors, including topography, soil properties, vegetation, and climatic conditions. These factors introduce variability in flow paths, storage zones, and velocities, which collectively shape the distribution of transit times. As a result, TTDs are inherently heterogeneous and challenging to quantify. Advances in analytical techniques, tracer studies, and modeling approaches have enhanced our ability to estimate TTDs, yet significant uncertainties remain due to data limitations and the simplifications inherent in models.



Description of the method
~~~~~~~~~~~~~~~~~~~~~~~~~

Our model is easily extensible to incorporate additional features. A common example is the inclusion of a **time variable**, which enables the model to learn potential temporal variations in catchment behavior.

## Training

To train the **WATRES** model, the following inputs are required:

- Tracer input and output concentrations  
- Precipitation  
- Streamflow  
- Potential evapotranspiration (PET)  

PET can be estimated using external tools such as the **Pyeto** package. All input data must share a **consistent time resolution** — this can be arbitrarily chosen (e.g., hourly, daily, or weekly), as long as it's applied uniformly across all inputs.

## Inference

Once trained, the WATRES model can infer **transit time distributions** using only **flux data** (precipitation and streamflow); **tracer data is not required during inference**.

Applications of the method
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once trained on a given site for which at least precipitation and streamflow time series are available, one can predict the transit time distributions at any time point. Therefore, our model paves the way to many application to draw hydroglogical insights on the way different catchments behave. 
