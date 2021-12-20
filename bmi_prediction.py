"""
This script is utilized to calculate insuarance quote according to bussiness rules.
It utilises model and scaler built in the jupyter notebook after experimentation to first predict BMI using height and weight.
After predicting BMI it uses age and gender according to the bussiness rules to determine the insurance quote.

The script has two interfaces within it-:

1. run_single_insurance_qoute_provider - Utility which utilises InsuranceQuoteProvider class and its provide_quote_single method
                                         to get insurance for a single user.
2. provide_quote_batch_multiprocess - Utility which utilises multiporcessing on CPU to get insurace quote for a list of users.

Classes in the script:
    1. InsuranceQuoteProvider
        a. provide_quote_single - Provides insurance quote for a user
        b. pred_BMI_single - predicts BMI using height and weight of user
        c. calculate_BMI_single - calculates BMI using matematical formula and height and weight of user
"""

import logging
import multiprocessing as mp
import pandas as pd
import pickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
logger.addHandler(console)

LIN_REG_MODEL_PATH = 'models/linear_reg.pickle'
SCALER_PATH = 'models/scaler.pickle'
READ_MODE = 'rb'
FT_TO_M = 0.3048
IN_TO_M = 0.0254


class InsuranceQuoteProvider:
    """
    Class utilized to contain methods to generate, validate, and persist aspects of the RPA workflow..

    Methods:
      - _load_model()
          Loads model from model path
      - _load_scaler()
          Load scaler from scaler path
      - _convert_pound_to_kgs()
          Converts weight from pounds to kg
      - _convert_ht_to_m()
          Converts height from input format to meters
      - calculate_BMI_single()
          Calculates BMI using matematical formula and height and weight of user
      - pred_BMI_single()
          Predicts BMI using height and weight of user
      - provide_quote_single()
          Provides insurance quote for a user
    """

    def __init__(self, lin_reg_model_path = None, scaler_path = None):
        self.lin_reg_model_path = LIN_REG_MODEL_PATH if lin_reg_model_path is None else lin_reg_model_path
        self.scaler_path = SCALER_PATH if scaler_path is None else scaler_path

    def _load_model(self):
        """
        Loads model from model path
        """
        return pickle.load(open(self.lin_reg_model_path, READ_MODE))

    def _load_scaler(self):
        """
        Load scaler from scaler path
        """
        return pickle.load(open(self.scaler_path, READ_MODE))

    def _convert_pound_to_kgs(self, weight_lbs):
        """
        Converts weight from pounds to kg
        :param: weight_lbs: weight in pounds
        :return: weight_kg: weight in kgs
        """
        weight_kg = 0.453592*weight_lbs
        return weight_kg

    def _convert_ht_to_m(self, height_raw):
        """
        Converts height from input format to meters
        :param: height_raw: height in input format
        :return: ht_m: height in meters
        """
        ht_modified = str(height_raw/100)
        ht_ft, ht_in = ht_modified.split('.', 1)
        ht_ft = int(ht_ft)
        ht_in = int(ht_in)
        ht_m = ht_ft*FT_TO_M + ht_in*IN_TO_M
        return ht_m

    def calculate_BMI_single(self, height_raw, weight_lbs):
        """
        Calculates BMI using matematical formula and height and weight of user
        :param: height_raw: height in input format
        :param: weight_lbs: weight in pounds
        :return: BMI_calc: calculated BMI
        """
        logger.info(f'Calculating BMI for single input using formula. Input Height: {height_raw/100} ft. Input weight: {weight_lbs} pounds.')
        weight = self._convert_pound_to_kgs(weight_lbs)
        height = self._convert_ht_to_m(height_raw)
        logger.info(f'Calculating BMI for single input using formula. Input Height: {height} m. Input weight: {weight} kg')
        BMI_calc = (weight)/(height**2)
        logger.info(f'Calculated BMI: {BMI_calc}')
        return BMI_calc

    def pred_BMI_single(self, height_raw, weight_lbs):
        """
        Predicts BMI using height and weight of user
        :param: height_raw: height in input format
        :param: weight_lbs: weight in pounds
        :return: BMI_pred: predicted BMI
        """
        logger.info(f'Predicting BMI for single input. Input Height: {height_raw/100} ft. Input weight: {weight_lbs} pounds.')
        weight = self._convert_pound_to_kgs(weight_lbs)
        height = self._convert_ht_to_m(height_raw)
        logger.info(f'Predicting BMI for single input. Input Height: {height} m. Input weight: {weight} kg')
        loaded_model = self._load_model()
        loaded_scaler = self._load_scaler()
        df = pd.DataFrame({'Ht_m': [height], 'Wt_kg': [weight]})
        df_scaled = loaded_scaler.transform(df)
        BMI_pred = loaded_model.predict(df_scaled).tolist()[0]
        logger.info(f'Predicted BMI: {BMI_pred}')
        return BMI_pred

    def provide_quote_single(self, age, gender, height, weight):
        """
        Provides insurance quote for a user
        :param: age: age in years
        :param: gender: 'Male'/'Female'
        :param: height: height in input format
        :param: weight: weight in pounds
        :return: insurance_quote: insurance quote in dollars
        """
        BMI_pred = self.pred_BMI_single(height, weight)
        insurance_quote = None
        if 18 <= age <= 39 and (17.49 >= BMI_pred or BMI_pred >= 38.5):
            insurance_quote = 750
            logger.info(f'Insurace quote {insurance_quote} . Age is between 18 to 39 and {BMI_pred} is either less than 17.49 or greater than 38.5')
        elif 40 <= age <= 59 and (18.49 >= BMI_pred or BMI_pred >= 38.5):
            insurance_quote = 1000
            logger.info(f'Insurace quote {insurance_quote} . Age is between 40 to 59 and {BMI_pred} is either less than 18.49 or greater than 38.5')
        elif age >= 60 and (18.49 >= BMI_pred or BMI_pred >= 45.5):
            insurance_quote = 2000
            logger.info(f'Insurace quote {insurance_quote} . Age is greater than 60 and {BMI_pred} is either less than 18.49 or greater than 45.5')
        else:
            insurance_quote = 500
            logger.info(f'Insurace quote {insurance_quote} . {BMI_pred} is in right range')

        if gender == 'Female':
            insurance_quote = insurance_quote - 0.1*insurance_quote
            logger.info(f'Gender discount applied. New Insurace quote {insurance_quote} .')
        
        return insurance_quote


def run_single_insurance_qoute_provider(age, gender, height, weight):
    """
    Utility which utilises InsuranceQuoteProvider class and its provide_quote_single method
    to get insurance for a single user. Method first predicts BMI using height and weight
    using the model build in Jupyter notebook and then applies bussiness logic to create a insurance quote
    :param: age: age in years
    :param: gender: 'Male'/'Female'
    :param: height: height in input format
    :param: weight: weight in pounds
    :return: insurance_quote: insurance quote in dollars
    """
    insurance_quote = InsuranceQuoteProvider().provide_quote_single(age, gender, height, weight)
    return insurance_quote


def provide_quote_batch_multiprocess(batch_inputs, multiprocessing_cores=None):
    """
    Utility which utilises multiporcessing on CPU to get insurace quote for a list of users.
    :param: batch_inputs: List of inputs in form [(age1, gender1, height1, weight1), (age2, gender2, height2, weight2), ...]
    :return: result_list: List of insurance quotes for each input
    """
    if multiprocessing_cores is None:
        pool = mp.Pool()
    else:
        pool = mp.Pool(processes=multiprocessing_cores)
    
    result_list = pool.starmap(run_single_insurance_qoute_provider, [(age, gender, height, weight) for age, gender, height, weight in batch_inputs])
    return result_list

        
if __name__ == '__main__':
    obese_weight_lbs = 300
    my_weight_lbs = 172
    my_height_raw = 509

    batch_inputs = [(15, 'Female', my_height_raw, obese_weight_lbs),
                    (25, 'Female', my_height_raw, obese_weight_lbs),
                    (45, 'Female', my_height_raw, obese_weight_lbs),
                    (65, 'Female', my_height_raw, obese_weight_lbs),
                    (24, 'Male', my_height_raw, my_weight_lbs)]

    # InsuranceQuoteProvider().pred_BMI_single(my_height_raw, my_weight_lbs)
    # InsuranceQuoteProvider().calculate_BMI_single(my_height_raw, my_weight_lbs)
    # InsuranceQuoteProvider().provide_quote_single(65, 'Female', my_height_raw, my_weight_lbs)
    result_list = provide_quote_batch_multiprocess(batch_inputs)
    logger.info(f'Result list: {result_list}')
