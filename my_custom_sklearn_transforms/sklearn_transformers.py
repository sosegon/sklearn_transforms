from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data.drop(labels=self.columns, axis='columns')
    
class ScaleColumns(BaseEstimator, TransformerMixin):
    def __init__(self, column_names_avoid):
        self.column_names_avoid = column_names_avoid
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        tr = StandardScaler(copy=True)
        
        data = X.copy()
        
        data_1 = data.drop(self.column_names_avoid, axis=1)
        
        tr_transformer = tr.fit(X=data_1)
        
        data_2 = tr_transformer.transform(X=data_1)
        
        data_3 = pd.DataFrame.from_records(
            data=data_2,
            columns=data_1.columns
        )
        
        data_4 = data[self.column_names_avoid]
        
        data_5 = data_4.join(data_3)
        
        return data_5

class Inputer(self):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        si = SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='most_frequent',  # la estrategia elegida es cambiar el valor faltante por una constante
            fill_value=0,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )

        si.fit(X)

        data = pd.DataFrame.from_records(
            data=si.transform(X),
            columns=X.columns
        )

        return data