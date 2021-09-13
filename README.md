# random-forest
stage two of the project

* rf_classification1.py classifies peptides into 2 types ('N' - scored 0, 'Y' - others)
  * data needed (have uploaded to Teams)
    * `feature_file = '../saved data/rfe2_204116/rfe2_dict_204116'`
    * `df_dataset_file = '../saved data/df_dataset'`
  * overfitting on training set too much, almost can't predict peptides from 'Y'

* rf_regressor6.py are for feature eliminating

  * data needed:
    * `df_dataset_file = '../saved data/df_dataset'`

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

  