'''
Download data from google drive using basic python packages.

MOSTLY BROKEN - DO NOT ATTEMPT TO USE AT THIS TIME
'''

# Darien Morrow - darienmorrow@gmail.com - dmorrow3@wisc.edu
# Blaise Thompson - blaise@untzag.com


### import ####################################################################


import os
import urllib


### helper methods ############################################################

def _authenticate(path):
    """
    Authenticate Google Drive API for usage in PyCMDS.
    
    Attributes
    ----------
    path : string
        Folder path of mycreds.txt file.
    
    NB: This function requires a Client_secrets.json file to be in the working directory.
    
    This function, once run, will open up a login window in a web browser.
    The user must then athenticate via email and password to authorize the
        API for usage with that particular account.
    Note that 'mycreds.txt' may just be an empty text file. This function will
        create the correct dictionary structure in the file upon completion. 
    """
        
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from pydrive.files import GoogleDriveFile                
        
    creds_path = os.path.join(path, 'mycreds.txt')
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(creds_path)  # try to load saved client credentials.
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()  # authenticate if credentials are not found.
    elif gauth.access_token_expired:
        gauth.Refresh()  # refresh credentials if they are expired.
    else:
        gauth.Authorize()  # initialize the saved credentials.
    gauth.SaveCredentialsFile(creds_path) # save the current credentials to a file




def filenamegetter(fileID):
    """
    Return the filename corresponding to a public Google Drive ID.
    
    Attributes
    ----------
    fileID : string    
        Unique Google Drive ID that may correspond to a folder if desired.        
        
    Returns
    -------
    string
        A string that is the human readable filename of the inputed file ID.
        If applicable the file extension of the file is included in the filename.
    """    

    urltoopen = ''.join(['https://drive.google.com/file/d/',fileID,'/view?pref=2&pli=1'])    
    print urltoopen    
    gdrivecontent0 = urllib.urlopen(urltoopen).readlines()  
    title = gdrivecontent0[0].split("<title>")[1].split("- Google Drive")[0]    
    return title

def filedownloader(fileID, savelocation):
    """
    Download file corresponding to a public Google Drive ID.
    
    Attributes
    ----------
    fileID : string    
        Unique Google Drive ID that may NOT correspond to a folder currently.        
    save location : string
        Valid, local file path to a directory in which to save file.

    Returns
    -------
    string
        A string that is the human readable filename of the inputed file ID.
        If applicable the file extension of the file is included in the filename.

    Examples
    --------
    gdrive_filedownloader('0B8z-JGr_8g4RWm0wb0tDWjZUcWc','/Users/darienmorrow/Desktop/')
    """     
 
    urllib.urlretrieve('https://docs.google.com/uc?export=download&id='+fileID ,savelocation+gdrive_filenamegetter(fileID))


def contentelucidator(fileID, titleYN, folderYN, contentsYN):
    """
    Return information about file or folder corresponding to a public Google Drive ID.
     
    Attributes
    ----------
    fileID : string    
        Unique Google Drive ID that may correspond to a folder.        
    titleYN : Boolean
        Toggles if function prints retrieved filename of fileID.
    folderYN : Boolean
        Toggles if function will return statement that states if fileID 
        corresponds to a file or folder.
    contentsYN : Boolean 
        Toggles if function will return all info retrieved from fileID URL.      

    Returns
    -------
    string
        Title and file extension of fileID.
    string
        Statement concerning identity of fileID:        
        'is not folder' or 'is folder'
    string
        Many lines of strings that are all the info hijacked from Google.

    Examples
    --------
    An example of a folder:    
    gdrive_contentelucidator('0B5XBhboKMJCTbFR4WUY0TG5QajA', True, True, True)

    An example of a file:
    gdrive_contentelucidator('0B8z-JGr_8g4RWm0wb0tDWjZUcWc', True, True, True)  
    """
    
    # /d/ is download statement
    urltoopen = ''.join(['https://drive.google.com/file/d/',fileID,'/view?pref=2&pli=1'])    
    gdrivecontent0 = urllib.urlopen(urltoopen).readlines()  
    title = gdrivecontent0[0].split("<title>")[1].split("- Google Drive")[0]    
    
    if titleYN == True:
        print title
    if folderYN == True:     
        if 'image' in gdrivecontent0[0]:
            print 'is not folder' 
        else:
            print 'is folder'
    if contentsYN == True:
        for element in gdrivecontent0:
            print element


### main download method ######################################################


def download(fileID, output_folder=None):
    url = ''.join(['https://drive.google.com/folderview?id=', fileID, '&usp=sharing'])    
    lines = urllib.urlopen(url).readlines()  
    title = lines[0].split("<title>")[1].split("</title>")[0]    
    # remove irrelevant lines
    for i in range(len(lines))[::-1]:
        if not lines[i][:3] == ',[,':
            lines.pop(i)
    return lines
    



### testing ###################################################################


if __name__ == '__main__':
    fileID = '0B5XBhboKMJCTYjRIQjlDVi1icEU'
    out = download(fileID)
    
    if False:
        #folder
        #gdrive_universaldownloader('0B5XBhboKMJCTbFR4WUY0TG5QajA')
        #file
        download('0B8z-JGr_8g4RWm0wb0tDWjZUcWc')
        download('0B5XBhboKMJCTbFR4WUY0TG5QajA')
        contentelucidator('0B5XBhboKMJCTbFR4WUY0TG5QajA', True, True, True)
        print filenamegetter('0B5XBhboKMJCTbFR4WUY0TG5QajA')
        filedownloader('0B8z-JGr_8g4RWm0wb0tDWjZUcWc','/Users/darienmorrow/Desktop/')
