# Adult dataset

`adult.data` is the supplied Adult data file, renamed without modifying its bytes. It contains 32,561 nonempty records and 14 predictors plus the income label. The separate UCI `adult.test` file is not included or used. This project creates its own stratified 80/20 split from the included file.

Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

Source: https://archive.ics.uci.edu/dataset/2/adult

UCI lists the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Dataset attribution applies to the data and is separate from licensing of project code and reports. No additional code license is granted by this repository.

SHA-256 of the included file: `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`

Missing values appear as `?`. The analysis keeps these rows using the categorical value `Unknown`. Recorded `sex` categories are Female and Male; they are not a comprehensive representation of gender identity.
