# random-forest
stage two of the project

* rf_classification1.py classified peptides into 2 types ('N' - scored 0, 'Y' - others)

* rf_regressor6.py are for feature eliminating

  * Try if you have `df_dataset_file = '../saved data/df_dataset'`

* saved data in Teams Group File

	- **df_f** is the dataframe of chemical features ([374489 rows x 5270 columns])

	- **df_s** is the dataframe of scores ([576509 rows x 1 columns])

	* **df_dataset** is the dataframe of the dataset needed for training and predicting. It is made by droping the duplicate columns after inner joining df_f and df_s. ([340193 rows x 1908 columns])

  	You can use this method to load them.	

```python
  def load_object(filename):
      with open(filename, 'rb') as inp:
          data = pickle.load(inp)
      return data
```

  