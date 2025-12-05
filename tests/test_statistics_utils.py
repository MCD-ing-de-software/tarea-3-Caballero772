import numpy as np
import numpy.testing as npt
import unittest

from src.statistics_utils import StatisticsUtils


class TestStatisticsUtils(unittest.TestCase):
    """Test suite for StatisticsUtils class."""

    def test_example_moving_average_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para comparar arrays de NumPy.
        
        Este test demuestra cómo usar numpy.testing.assert_allclose() para comparar
        arrays de NumPy con tolerancia para errores de punto flotante, lo cual es
        esencial cuando trabajamos con operaciones numéricas.
        """
        utils = StatisticsUtils()
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = utils.moving_average(arr, window=3)
        
        # Valores esperados para media móvil con window=3
        expected = np.array([2.0, 3.0, 4.0])
        
        # Usar numpy.testing.assert_allclose() para comparar arrays de NumPy
        # Esto maneja correctamente errores de punto flotante con tolerancia
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_example_min_max_scale_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para verificar transformaciones numéricas.
        
        Este test demuestra cómo usar numpy.testing.assert_allclose() para verificar
        que una transformación numérica produce los resultados correctos en todo el array,
        considerando errores de punto flotante en cálculos matemáticos.
        """
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0]
        result = utils.min_max_scale(arr)
        
        # Valores esperados después de min-max scaling: (x - min) / (max - min)
        # min=10, max=40, range=30
        # [10->0.0, 20->0.333..., 30->0.666..., 40->1.0]
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        
        # Usar numpy.testing.assert_allclose() para comparar arrays numéricos
        # La tolerancia relativa y absoluta permite errores pequeños de punto flotante
        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_moving_average_basic_case(self):
        """Test que verifica que el método moving_average calcula correctamente la media móvil
        de una secuencia numérica para un caso básico."""
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0, 50.0]
        window = 3
        
        expected = np.array([20.0, 30.0, 40.0])
        result = utils.moving_average(arr, window=window)
        
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
        self.assertEqual(result.shape, expected.shape)

    def test_moving_average_raises_for_invalid_window(self):
        """Test que verifica que el método moving_average lanza un ValueError cuando
        se proporciona una ventana (window) inválida."""
        utils = StatisticsUtils()
        arr = [1, 2, 3]
        
        # Caso 1: window=0 (valor no positivo)
        with self.assertRaises(ValueError) as context:
            utils.moving_average(arr, window=0)
        self.assertIn("window must be a positive integer", str(context.exception))
        
        # Caso2: window mayor que la longitud del array
        with self.assertRaises(ValueError) as context:
            utils.moving_average(arr, window=4)
        self.assertIn("window must not be larger than the array size", str(context.exception))

    def test_moving_average_only_accepts_1d_sequences(self):
        """Test que verifica que el método moving_average lanza un ValueError cuando
        se llama con una secuencia multidimensional."""
        utils = StatisticsUtils()
        arr_2d = [[1, 2], [3, 4]]
        
        with self.assertRaises(ValueError) as context:
            utils.moving_average(arr_2d, window=2)
            
        self.assertIn("moving_average only supports 1D sequences", str(context.exception))

    def test_zscore_has_mean_zero_and_unit_std(self):
        """Test que verifica que el método zscore calcula correctamente los z-scores
        de una secuencia numérica, comprobando que el resultado tiene media cero y
        desviación estándar unitaria."""
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        z_scores = utils.zscore(arr)
        
        self.assertAlmostEqual(z_scores.mean(), 0.0, places=7)
        self.assertAlmostEqual(z_scores.std(ddof=0), 1.0, places=7)

    def test_zscore_raises_for_zero_std(self):
        """Test que verifica que el método zscore lanza un ValueError cuando
        se llama con una secuencia que tiene desviación estándar cero
        (todos los valores son iguales)."""
        utils = StatisticsUtils()
        arr_constant = [5, 5, 5, 5]
        
        with self.assertRaises(ValueError) as context:
            utils.zscore(arr_constant)
            
        self.assertIn("Standard deviation is zero; z-scores are undefined", str(context.exception))

    def test_min_max_scale_maps_to_zero_one_range(self):
        """Test que verifica que el método min_max_scale escala correctamente una secuencia
        numérica al rango [0, 1], donde el valor mínimo se mapea a 0 y el máximo a 1."""
        utils = StatisticsUtils()
        arr = [2.0, 4.0, 6.0, 8.0]

        expected = np.array([0.0, 1/3, 2/3, 1.0])
        
        scaled_arr = utils.min_max_scale(arr)
        
        npt.assert_allclose(scaled_arr, expected, rtol=1e-7, atol=1e-7)

        self.assertAlmostEqual(scaled_arr.min(), 0.0, places=7)
        
        self.assertAlmostEqual(scaled_arr.max(), 1.0, places=7)

    def test_min_max_scale_raises_for_constant_values(self):
        """Test que verifica que el método min_max_scale lanza un ValueError cuando
        se llama con una secuencia donde todos los valores son iguales (no hay variación)."""
        utils = StatisticsUtils()
        arr_constant = [3, 3, 3, 3]
        
        with self.assertRaises(ValueError) as context:
            utils.min_max_scale(arr_constant)
            
        self.assertIn("All values are equal; min-max scaling is undefined", str(context.exception))


if __name__ == "__main__":
    unittest.main()
