
""" I/O for ztfimg data. """

import os
import pandas
import numpy as np

LOCALSOURCE = os.getenv('ZTFDATA',"./Data/")
CALIBRATOR_PATH = os.path.join(LOCALSOURCE,"calibrator")
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

def get_test_image():
    """ returns the path to the test image and its mask. """
    sciimg = os.path.join(PACKAGE_PATH, "data/ztf_20200924431759_000655_zr_c13_o_q3_sciimg.fits")
    maskimg = os.path.join(PACKAGE_PATH, "data/ztf_20200924431759_000655_zr_c13_o_q3_mskimg.fits")
    return sciimg, maskimg

def get_nonlinearity_table(date):
    from .utils.tools import ccdid_qid_to_rcid
    
    NL_DICT = {'2018' : './data/linearity_coeffs_201803.txt', 
               '2019': './data/linearity_coeffs_201910.txt', 
               '2020' : './data/linearity_coeffs_202004.txt'}
    
    if date < np.datetime64('2019-10-23', 'D') : 
        filepath = NL_DICT['2018']
        
    elif date >= np.datetime64('2019-10-23', 'D') and date < np.datetime64('2020-04-28', 'D'):
        filepath = NL_DICT['2019']
        
    else : 
        filepath = NL_DICT['2020']

    NONLINEARITY_FILE = os.path.join(PACKAGE_PATH, filepath)
    nl_table = pandas.read_csv(NONLINEARITY_FILE, comment='#', header=None, sep=r'\s+', usecols=[0, 1, 2, 3, 4],
                                      names=["ccdid", "ampname", "qid", "a", "b"])
    nl_table["qid"] += 1 # qid in the file is actually AMP_ID that starts at 0, while qid starts at 1.
    nl_table["rcid"] = ccdid_qid_to_rcid(nl_table["ccdid"].astype("int"), nl_table["qid"].astype("int"))
    return nl_table.set_index("rcid").sort_index()
