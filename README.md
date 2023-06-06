SOM-QE is a software tool that can be used to detect differences within data sets. For example, it can be applied to tell the differences between two sets of medical images taken from the same patient at different times - possibly two consecutive clinical visits. Thus, it can assist the radiologists/surgeons in determining the effects of their prescriptions on the patient between the duration of the two clinical visits.

It is based on the artificial neural network algorithm called Self-Organizing Map developed by Teuvo Kohonen. You can read more about it in his [book](http://docs.unigrafia.fi/publications/kohonen_teuvo/MATLAB_implementations_and_applications_of_the_self_organizing_map.pdf), or in one of his numerous publications available online.

In this demonstration of the working of SOM-QE, an implementation of SOM called [MiniSOM](https://github.com/JustGlowing/minisom), is used.

You can read on how we used SOM-QE to tell the difference between two sets of images, obtained from a patient with a sprained from this [open source paper](https://www.sciencedirect.com/science/article/pii/S2352914817300059), for details. 

In this example, SOM-QE is used to process satelite images of Lake Mead - a water reservoir formed by the Hoover Dam on the Colorado River in the Southwestern United States - to determine rising/falling water levels. Sections of the image on water levels were extracted from the year 1984 to 2011. The SOM-QE values obtained indicate the changing water levels of the lake over the years, which corresponds with the findings of other measurements by organization like [Bureau of Reclamation](https://www.usbr.gov/lc/region/g4000/lakemead_line.pdf) as accessed on 3rd June 2023.
