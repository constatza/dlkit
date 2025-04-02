import mlflow
logged_model = 'runs:/493e5786e5bf46158b9204b0026a4fa0/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))