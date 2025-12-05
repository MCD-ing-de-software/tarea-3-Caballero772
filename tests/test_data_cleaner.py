import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    df = pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )

    df["name"] = df["name"].astype(pd.StringDtype()) 
    
    return df


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que el método drop_invalid_rows elimina correctamente las filas
        que contienen valores faltantes (NaN o None) en las columnas especificadas."""
        cleaner = DataCleaner()
        df = make_sample_df()
        
        result_df = cleaner.drop_invalid_rows(df, ["name", "age"])
        
        self.assertLess(len(result_df), len(df))
        self.assertEqual(len(result_df), 2)
        
        self.assertEqual(result_df["name"].isna().sum(), 0)
        self.assertEqual(result_df["age"].isna().sum(), 0)
        
        pdt.assert_index_equal(result_df.index, pd.Index([0, 3]))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que el método drop_invalid_rows lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame."""
        cleaner = DataCleaner()
        df = make_sample_df()
        
        with self.assertRaises(KeyError) as context:
            cleaner.drop_invalid_rows(df, ["age", "does_not_exist"])
            
        self.assertIn("does_not_exist", str(context.exception))

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica que el método trim_strings elimina correctamente los espacios
        en blanco al inicio y final de los valores en las columnas especificadas, sin modificar
        el DataFrame original ni las columnas no especificadas."""
        cleaner = DataCleaner()
        df_original = make_sample_df()
        
        df_copy_for_immutability = df_original.copy()
        
        result_df = cleaner.trim_strings(df_original, ["name"])
        
        self.assertEqual(result_df.loc[0, "name"], "Alice")
        self.assertEqual(result_df.loc[3, "name"], "Carol")
        
        pdt.assert_frame_equal(df_original, df_copy_for_immutability)
        self.assertEqual(df_original.loc[0, "name"], " Alice ") 

        pdt.assert_series_equal(result_df["city"], df_original["city"], check_names=True)
        self.assertEqual(result_df.loc[0, "city"], "SCL")

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que el método trim_strings lanza un TypeError cuando
        se llama con una columna que no es de tipo string."""
        cleaner = DataCleaner()
        df = make_sample_df()
        
        with self.assertRaises(TypeError) as context:
            cleaner.trim_strings(df, ["age"])
            
        self.assertIn("Columns are not string dtype", str(context.exception))

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica que el método remove_outliers_iqr elimina correctamente los
        valores extremos (outliers) de una columna numérica usando el método del rango
        intercuartílico (IQR)."""
        cleaner = DataCleaner()
        df = make_sample_df() 
        
        df_outlier = pd.DataFrame({'value': [10, 20, 30, 40, 50, 150]})
             
        result_df = cleaner.remove_outliers_iqr(df_outlier, 'value', factor=1.5)
        
        self.assertNotIn(150, result_df['value'].values)
        
        self.assertIn(50, result_df['value'].values)
        
        self.assertEqual(len(result_df), 5)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame."""
        cleaner = DataCleaner()
        df = make_sample_df()
        
        with self.assertRaises(KeyError) as context:
            cleaner.remove_outliers_iqr(df, "salary")
            
        self.assertIn("Column 'salary' not found in DataFrame", str(context.exception))

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un TypeError cuando
        se llama con una columna que no es de tipo numérico."""
        cleaner = DataCleaner()
        df = make_sample_df()
        
        with self.assertRaises(TypeError) as context:
            cleaner.remove_outliers_iqr(df, "city")
            
        self.assertIn("Column 'city' must be numeric to compute IQR", str(context.exception))


if __name__ == "__main__":
    unittest.main()
