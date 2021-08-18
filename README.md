# random-forest
stage two of the project

* saved data/df_s and saved data/df_f are two dataframe stores info of scores and features. 

  You can use this method to load them.

  ```python
  def load_object(filename):
      with open(filename, 'rb') as inp:
          data = pickle.load(inp)
      return data
  ```

  