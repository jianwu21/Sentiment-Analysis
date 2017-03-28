""" Utilities for the sentiment analysis project
Includes:
    check_download_data: Checks and downloads data...
"""
import os
import urllib.request
import traceback
import zipfile


URL = "http://2.110.57.134/LangProc2/scaledata_TRAIN.zip"


def download_data(override=False, path="./data", url=URL):
    """ Downloads and extracts a zip file from the web.
    Arguments:
        override (bool): Override the data even if folders exist.
            Defaults to True
        path (str): Path to put the data. Defaults to './data'
        url (str): URL to retrieve. Defaults to Movie Review Dataset
            on Peter's server
    Returns:
        exit code (int): 0 if success, 1 if any problem
    """
    checks = (os.path.exists(path + '/scaledata'),
              os.path.exists(path + '/scale_whole_review'),
              not override)

    if all(checks):
        print("Data seem to be available, to force redownloading "
              "use 'override=True'")
        return 0

    try:
        print("Downloading zip file, please be patient...", end='', flush=True)
        filehandle, _ = urllib.request.urlretrieve(url)
        print("\rExtracting zip file...", end='', flush=True)
        zip_object = zipfile.ZipFile(filehandle, 'r')
        zip_object.extractall(path)
        print("\nDone!")
        return 0
    except:
        print("An error occured. DEBUG INFO:")
        print(traceback.format_exc())
        return 1
