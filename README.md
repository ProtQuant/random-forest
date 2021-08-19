# random-forest
stage two of the project

* some saved data

	GitHub release has a limit of 2GB.

	The data below is uploaded on teams, in the group file.

	Random forest model will be uploaded later.

  - **df_f** is the dataframe of chenical features ([374489 rows x 5270 columns])

  - **df_s** is the dataframe of scores ([576509 rows x 1 columns])
  
  * **df_dataset** is the dataframe of the dataset needed for training and predicting. It is made by droping the duplicate columns after inner joining df_f and df_s. ([340193 rows x 1908 columns])
  
  	You can use this method to load them.	
  
```python
  def load_object(filename):
      with open(filename, 'rb') as inp:
          data = pickle.load(inp)
      return data
  ```
  
  