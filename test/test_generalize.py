import sys
import unittest
import pandas as pd
from anonymize.generalize import generalize_categorical, discretize
from anonymetrics.anonymetrics import calculate_k_anonymity


class TestGeneralize(unittest.TestCase):
    def test_discretize(self):
        # Define user data
        users = {
            'user_id': [1, 2, 3, 4, 5],
            'age': [1.5, 4.3, 7.9, 15.8, 30.0],
            'income': [40000, 40000, 60000, 45000, 90000]
        }

        df = pd.DataFrame(users)

        k = calculate_k_anonymity(df, qa_indices=[0])
        self.assertEqual(k, 1)

        discretize(df, 1, 5.0)

        # Expected discretized ages
        expected_ages = [(0.0, 4.0), (0.0, 4.0), (5.0, 9.0), (15.0, 19.0), (30.0, 34.0)]

        # Check if the 'age' column has been discretized correctly
        self.assertEqual(df['age'].tolist(), expected_ages)

    def test_generalize_categorical(self):
        # Example dataframe
        df = pd.DataFrame({
            'img': [1, 2, 3, 4, 5],
            'Label 1': ['foo', 'bar', 'cat', 'dog', 'eagle'],
            'Label 2': ['foo', 'bar', 'cat', 'dog', 'bird']
        })

        k = calculate_k_anonymity(df, qa_indices=[1, 2])
        self.assertEqual(k, 1)

        # Call the function to generalize categorical attributes
        generalize_categorical(df, [1, 2], ['cat', 'dog', 'bird', 'eagle'])
        generalize_categorical(df, [1, 2], ['foo', 'bar'])

        k = calculate_k_anonymity(df, qa_indices=[1, 2])
        self.assertEqual(k, 2)


if __name__ == '__main__':
    unittest.main()
