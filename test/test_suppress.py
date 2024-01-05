import unittest
import pandas as pd
from anonymize.suppress import suppress_categorical, suppress_float, remove_groups
from anonymetrics.anonymetrics import get_groups


class TestSuppress(unittest.TestCase):
    def test_suppress_float(self):
        # Define user data
        users = {
            'user_id': [1, 2, 3, 4, 5],
            'age': [23, 23, 40, 29, 45],
            'income': [40000, 40000, 60000, 45000, 90000]
        }

        # Create a DataFrame
        df = pd.DataFrame(users)

        # Add an attribute column with integer values
        df['credit_score'] = [720, 680, 590, 650, 780]

        suppress_float(df, 3)
        self.assertEqual(df["credit_score"][0], 684.0)
        self.assertEqual(df["credit_score"][3], 684.0)

    def test_suppress_categorical(self):
        # Set up a dataframe for testing
        self.df = pd.DataFrame({
            'A': ['cat', 'dog', 'dog', 'bird', 'bird'],
            'B': ['apple', 'orange', 'banana', 'apple', 'banana'],
            'C': ['red', 'blue', 'red', 'green', 'blue']
        })

        # Call the suppress_categorical function
        suppress_categorical(self.df, 'dog', 0)

        # Expected output
        expected_output = pd.DataFrame({
            'A': ['cat', frozenset({'bird', 'dog', 'cat'}), frozenset({'bird', 'dog', 'cat'}), 'bird', 'bird'],
            'B': ['apple', 'orange', 'banana', 'apple', 'banana'],
            'C': ['red', 'blue', 'red', 'green', 'blue']
        })

        # Assert that the output matches the expected output
        pd.testing.assert_frame_equal(self.df, expected_output)

        # Call the suppress_categorical function
        suppress_categorical(self.df, 'bird', 0)

        # Expected output
        expected_output = pd.DataFrame({
            'A': ['cat', frozenset({'bird', 'dog', 'cat'}), frozenset({'bird', 'dog', 'cat'}), frozenset({'bird', 'dog', 'cat'}), frozenset({'bird', 'dog', 'cat'})],
            'B': ['apple', 'orange', 'banana', 'apple', 'banana'],
            'C': ['red', 'blue', 'red', 'green', 'blue']
        })

        pd.testing.assert_frame_equal(self.df, expected_output)


    def test_remove_groups(self):
        # Create a DataFrame for testing
        df = pd.DataFrame({
            'A': ['apple', 'banana', 'cherry', 'banana', 'apple', 'apple'],
            'B': ['red', 'yellow', 'red', 'red', 'yellow', 'red'],
            'C': ['small', 'large', 'medium', 'large', 'small', 'medium']
        })

        # Test if groups with less than 2 records are removed correctly

        remove_groups(df, [0, 1], 2)

        expected_result = pd.DataFrame({
            'A': ['apple', 'apple'],
            'B': ['red', 'red'],
            'C': ['small', 'medium']
        })
        # print(df)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_result)


if __name__ == '__main__':
    unittest.main()
