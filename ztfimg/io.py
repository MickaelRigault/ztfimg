
""" I/O for ztfimg data. """

import os
import pandas

LOCALSOURCE = os.getenv('ZTFDATA',"./Data/")
CALIBRATOR_PATH = os.path.join(LOCALSOURCE,"calibrator")
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

def get_test_image():
    """ returns the path to the test image and its mask. """
    sciimg = os.path.join(PACKAGE_PATH, "data/ztf_20200924431759_000655_zr_c13_o_q3_sciimg.fits")
    maskimg = os.path.join(PACKAGE_PATH, "data/ztf_20200924431759_000655_zr_c13_o_q3_mskimg.fits")
    return sciimg, maskimg

def get_nonlinearity_table():
    """ get the nonlinearity table 
    
    Returns
    -------
    DataFrame
    """
    from .utils.tools import ccdid_qid_to_rcid

    NONLINEARITY_FILE = os.path.join(PACKAGE_PATH, "data/ccd_amp_coeff_v2.txt")
    nl_table = pandas.read_csv(NONLINEARITY_FILE, comment='#', header=None, sep='\s+', usecols=[0, 1, 2, 3, 4],
                                      names=["ccdid", "ampname", "qid", "a", "b"])
    nl_table["qid"] += 1 # qid in the file is actually AMP_ID that starts at 0, while qid starts at 1.
    nl_table["rcid"] = ccdid_qid_to_rcid(nl_table["ccdid"].astype("int"), nl_table["qid"].astype("int"))
    return nl_table.set_index("rcid").sort_index()

